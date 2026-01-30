import zmq
import msgpack
import msgpack_numpy as m
import time
import numpy as np

# Patch msgpack to handle numpy arrays
m.patch()

class ROSZmqBridge:
    def __init__(self, port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        print(f"Server started on port {port}")

    def run(self):
        while True:
            # 1. Receive action from RL Agent
            message = self.socket.recv()
            data = msgpack.unpackb(message)

            print("Received data:", data.keys())
            
            cmd_vel = data.get("cmd_vel")
            command = data.get("command")
            reset = data.get("reset")

            # 2. Logic to interface with ROS 
            observation = self.step_ros(cmd_vel)


            # 3. Send back the dictionary
            reply = {
                "observation": observation,  # Dummy observation
                "number_of_collisions": 0,
                "done": reset,
            }
            self.socket.send(msgpack.packb(reply))
    
    def step_gazebo(self, steps: int):
        # Placeholder for stepping the Gazebo simulation
        print(f"Stepping Gazebo for {steps} steps")
        time.sleep(0.01)  # Simulate time delay
    
    def publish_cmd_vel(self, cmd_vel):
        # Placeholder for publishing cmd_vel to ROS topic
        print("Publishing cmd_vel to ROS:", cmd_vel)
    
    def get_observation(self):
        observation = np.zeros((64, 64, 3))

        return observation  # Dummy state
    
    def step_ros(self, cmd_vel):

        self.publish_cmd_vel(cmd_vel)
        self.step_gazebo(50)

        observation = self.get_observation()

        return observation # Dummy state

if __name__ == "__main__":
    bridge = ROSZmqBridge()
    bridge.run()