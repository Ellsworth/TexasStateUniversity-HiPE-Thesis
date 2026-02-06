import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock

import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
import threading
import time
import subprocess
import os
import signal

from ros_gz_interfaces.srv import ControlWorld
from ros_gz_interfaces.msg import WorldControl
from ros_gz_interfaces.msg import WorldReset
from ros_gz_interfaces.srv import SetEntityPose

# Patch msgpack (standard boilerplate)
m.patch()

class MinimalBridge(Node):
    def __init__(self, port=5555):
        super().__init__('ros_zmq_bridge')

        # Add a thread-safe way to store the latest grid
        self.latest_grid = None
        self.grid_lock = threading.Lock()
        
        # Simulation time tracking
        self.current_sim_time = 0.0
        self.sim_time_lock = threading.Lock()

        # Grid dimensions
        self.grid_height = 65
        self.grid_width = 65

        # Wall distance tracking
        self.wall_distance = -1.0
        self.wall_angle = 0.0
        self.scan_lock = threading.Lock()

        # 1. ROS Setup
        self.cmd_vel_pub = self.create_publisher(Twist, '/marble_hd2/cmd_vel', 10)
        self.create_subscription(Int16MultiArray, '/local_grid', self.grid_callback, 10)
        self.create_subscription(LaserScan, '/marble_hd2/planar_laser/scan', self.scan_callback, 10)
        self.create_subscription(Clock, '/clock', self.clock_callback, 10)

        # 1.1 Gazebo Control Client
        self.world_control_client = self.create_client(ControlWorld, '/world/shapes/control')
        self.set_entity_pose_client = self.create_client(SetEntityPose, '/world/shapes/set_pose')
        # We don't block waiting for service here to avoid stalling startup if sim isn't ready

        # 2. ZMQ Setup - Renamed to avoid ROS 2 conflict
        self.zmq_context = zmq.Context() 
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(f"tcp://*:{port}")
        self.get_logger().info(f"ZMQ Server listening on port {port}")

        # 3. Start ZMQ in background thread
        self.zmq_thread = threading.Thread(target=self.zmq_loop, daemon=True)
        self.zmq_thread.start()

    def grid_callback(self, msg):
        """Convert ROS Int16MultiArray to a NumPy grid for RL"""
        try:
            # 1. Convert flat data to NumPy array
            flat_grid = np.array(msg.data, dtype=np.int16)
            
            # 2. Reshape to 2D (assuming row-major order from ROS)
            # If the lengths don't match, this will trigger the except block
            grid_2d = flat_grid.reshape((self.grid_height, self.grid_width))
            
            # 3. Store safely for the ZMQ thread
            with self.grid_lock:
                self.latest_grid = grid_2d
                
            self.get_logger().debug("Successfully updated local_grid observation")
            
        except ValueError as e:
            self.get_logger().error(f"Grid reshape failed: {e}. Check height/width dimensions.")

    def scan_callback(self, msg):
        """Process LaserScan to find closest obstacle"""
        # Filter out invalid ranges (infs/nans) and find min
        ranges = np.array(msg.ranges)
        
        # Create angles array
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Filter out self-collision (rear view). 
        # Assume angles > 2.5 rad or < -2.5 rad are the back of the robot
        # This corresponds to roughly +/- 143 degrees
        view_mask = np.abs(angles) < 2.5
        
        # Also filter invalid ranges (inf/nan)
        valid_mask = np.isfinite(ranges) & view_mask
        
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        with self.scan_lock:
            if valid_ranges.size > 0:
                min_idx = np.argmin(valid_ranges)
                self.wall_distance = float(valid_ranges[min_idx])
                self.wall_angle = float(valid_angles[min_idx])
            else:
                self.wall_distance = -1.0
                self.wall_angle = 0.0

    def clock_callback(self, msg):
        with self.sim_time_lock:
            self.current_sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def step_simulation(self, steps=1):
        """Step the simulation by N steps"""
        if not self.world_control_client.service_is_ready():
            self.get_logger().warn("World Control service not ready! Ignoring step request.")
            return False

        req = ControlWorld.Request()
        req.world_control.multi_step = steps
        req.world_control.pause = True # Ensure we pause after stepping
        
        self.get_logger().debug(f"Stepping simulation by {steps} steps")

        # Capture start time
        with self.sim_time_lock:
            start_time = self.current_sim_time

        # Target time (assuming 1ms step size, common for Gazebo/ROS2)
        # TODO: Ideally should seek step size from sim params, but 0.001 is a safe standard assumption for Gazebo Classics/Ignition default
        step_size_sec = 0.001 
        target_time = start_time + (steps * step_size_sec)

        # Synchronous call to trigger steps
        result = self.world_control_client.call(req)
        
        if not result.success:
            self.get_logger().error("World control service returned failure!")
            return False

        # Wait until sim time reaches target (or timeout)
        timeout_sec = 2.0 # Reduced from 10.0s. If it takes >2s to step 0.1s sim time, something is wrong.
        wait_start_wall = time.time()
        
        # Epsilon for float comparison
        epsilon = 1e-5 

        while True:
            with self.sim_time_lock:
                current = self.current_sim_time
            
            if current >= target_time - epsilon:
                break
            
            # Check timeout
            if time.time() - wait_start_wall > timeout_sec:
                self.get_logger().warn(
                    f"Timeout waiting for simulation to step! "
                    f"Start: {start_time:.4f}, Current: {current:.4f}, "
                    f"Target: {target_time:.4f} (Delta: {current - start_time:.4f})"
                )
                break
                
            time.sleep(0.001) # Small sleep checking loop

        # Log detailed info if it took significantly longer than expected
        elapsed = time.time() - wait_start_wall
        if elapsed > 0.5:
             self.get_logger().warn(f"Step took {elapsed:.2f}s wall time (Target Sim Delta: {target_time - start_time:.4f})")
        else:
             self.get_logger().debug(f"Step completed. Sim Time: {start_time:.3f} -> {current:.3f}")
             
        return True

    def restart_cartographer(self):
        # Find the process ID of the cartographer node
        # We use 'pgrep' to find the PID based on the executable name
        try:
            pid = subprocess.check_output(["pgrep", "-f", "cartographer_node"]).decode().strip()
            if pid:
                # Send SIGTERM (or SIGKILL if it's being stubborn)
                # Handle multiple PIDs if any
                for p in pid.splitlines():
                     self.get_logger().info(f"Restarting Cartographer (PID: {p})...")
                     os.kill(int(p), signal.SIGTERM)
        except subprocess.CalledProcessError:
            self.get_logger().warn("Cartographer node not found. Is it running?")

    def zmq_loop(self):
        while rclpy.ok():
            received = False
            try:
                raw_msg = self.zmq_socket.recv()
                received = True
                data = msgpack.unpackb(raw_msg)
                
                self.get_logger().debug(f"Got message: {data}")

                # 0. Handle Simulation Control
                if "reset" in data and data["reset"]:
                    if self.set_entity_pose_client.service_is_ready():
                        req = SetEntityPose.Request()
                        req.entity.name = 'marble_hd2'
                        req.pose.position.x = 5.0
                        req.pose.position.y = 30.0
                        req.pose.position.z = 8.0
                        req.pose.orientation.x = 0.0
                        req.pose.orientation.y = 0.0
                        req.pose.orientation.z = 0.0
                        req.pose.orientation.w = 1.0

                        future = self.set_entity_pose_client.call_async(req)
                        # Since we are in a separate thread from rclpy.spin, we can wait for the future
                        # However, call_async + wait_for_future is safer or just use atomic call if spin is handling it?
                        # Since rclpy.spin(node) is running in main thread, and we are in zmq_thread:
                        # We cannot use self.set_entity_pose_client.call(req) because that might deadlock if valid spin is needed?
                        # Actually, call() inside a callbacks usually deadlocks, but here we are in a separate thread.
                        # But safely, we can just use call() if the executor handles it.
                        # Given the pattern in step_simulation uses call(), we will stick to that.
                        
                        try:
                           res = self.set_entity_pose_client.call(req)
                           if res.success:
                               self.get_logger().info("Successfully reset robot pose via /world/shapes/set_pose")
                               # Restart Cartographer
                               self.restart_cartographer()
                           else:
                               self.get_logger().error("Failed to reset robot pose")
                        except Exception as e:
                            self.get_logger().error(f"Service call failed: {e}")
                    else:
                        self.get_logger().warn("SetEntityPose service not ready")

                if "step" in data:
                    steps = int(data["step"])
                    self.step_simulation(steps)

                
                # Extract cmd_vel (comes in as a numpy array from msgpack_numpy)
                cmd = data.get("cmd_vel")

                if cmd is not None:
                    # 1. Create the Twist message instance
                    twist = Twist()

                    # 2. Assign the values
                    # cmd[0] = Linear X (forward/backward)
                    # cmd[1] = Angular Z (steering/yaw)
                    twist.linear.x = float(cmd[0])
                    twist.angular.z = float(cmd[1])

                    # 3. Explicitly zero out others (optional, as Twist defaults to 0.0)
                    twist.linear.y = 0.0
                    twist.linear.z = 0.0
                    twist.angular.x = 0.0
                    twist.angular.y = 0.0

                    # 4. Publish to the ROS 2 topic
                    self.cmd_vel_pub.publish(twist)
                    
                    self.get_logger().debug(f"Published to /cmd_vel: Linear={twist.linear.x}, Angular={twist.angular.z}")

                # Retrieve the latest grid safely
                with self.grid_lock:
                    obs = self.latest_grid if self.latest_grid is not None else []
                
                with self.scan_lock:
                    current_wall_dist = self.wall_distance
                    current_wall_angle = self.wall_angle

                # Send back the actual observation
                # msgpack_numpy handles the conversion of the NumPy array automatically
                reply = {
                    "status": "ok", 
                    "observation": obs,
                    "wall_distance": current_wall_dist,
                    "wall_angle": current_wall_angle
                }
                self.zmq_socket.send(msgpack.packb(reply))

            except Exception as e:
                self.get_logger().error(f"ZMQ Error: {e}")
                if received:
                    try:
                        # Attempt to send error reply to reset REP state
                        self.zmq_socket.send(msgpack.packb({"status": "error", "msg": str(e)}))
                    except Exception as send_e:
                         self.get_logger().error(f"Critical ZMQ State Error: {send_e}")

def main(args=None):
    rclpy.init(args=args)
    node = MinimalBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()