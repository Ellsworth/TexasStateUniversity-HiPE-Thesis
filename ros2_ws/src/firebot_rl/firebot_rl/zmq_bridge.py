import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16MultiArray

import zmq
import msgpack
import msgpack_numpy as m
import numpy as np
import threading

# Patch msgpack (standard boilerplate)
m.patch()

class MinimalBridge(Node):
    def __init__(self, port=5555):
        super().__init__('ros_zmq_bridge')

        # Add a thread-safe way to store the latest grid
        self.latest_grid = None
        self.grid_lock = threading.Lock()
        
        # Grid dimensions
        self.grid_height = 65
        self.grid_width = 65

        # 1. ROS Setup
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Int16MultiArray, '/local_grid', self.grid_callback, 10)

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

    def zmq_loop(self):
        while rclpy.ok():
            try:
                raw_msg = self.zmq_socket.recv()
                data = msgpack.unpackb(raw_msg)
                
                self.get_logger().info(f"Got message: {data}")

                
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
                    
                    self.get_logger().info(f"Published to /cmd_vel: Linear={twist.linear.x}, Angular={twist.angular.z}")

                # Retrieve the latest grid safely
                with self.grid_lock:
                    obs = self.latest_grid if self.latest_grid is not None else []

                # Send back the actual observation
                # msgpack_numpy handles the conversion of the NumPy array automatically
                reply = {
                    "status": "ok", 
                    "observation": obs
                }
                self.zmq_socket.send(msgpack.packb(reply))

            except Exception as e:
                self.get_logger().error(f"ZMQ Error: {e}")

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