import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16MultiArray

import zmq
import msgpack
import msgpack_numpy as m
import threading

# Patch msgpack (standard boilerplate)
m.patch()

class MinimalBridge(Node):
    def __init__(self, port=5555):
        super().__init__('ros_zmq_bridge')

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
        """Just log when we get a message from ROS"""
        length = len(msg.data)
        self.get_logger().deb(f"ROS: Received /local_grid data (length: {length})")

    def zmq_loop(self):
        """Blocking loop to handle ZMQ messages"""
        while rclpy.ok():
            try:
                # Use the renamed socket
                raw_msg = self.zmq_socket.recv()
                data = msgpack.unpackb(raw_msg)
                
                # ... (rest of your logic) ...

                # Send Dummy Reply
                reply = {"status": "ok", "observation": []}
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