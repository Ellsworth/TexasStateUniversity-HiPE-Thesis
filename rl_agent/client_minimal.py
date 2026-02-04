import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
import time

m.patch()

class RLZmqClient:
    def __init__(self, ip="127.0.0.1", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip}:{port}")

    def step(self, cmd_vel, reset: bool):
        # Create the dictionary payload
        payload = {
            "command": "step",
            "cmd_vel": cmd_vel,
            "reset": reset
        }

        print(f"Sending cmd_vel: {cmd_vel}, reset: {reset}")
        
        # Send and wait for reply
        self.socket.send(msgpack.packb(payload))
        message = self.socket.recv()
        
        return msgpack.unpackb(message)

# Example RL Loop
if __name__ == "__main__":
    env_client = RLZmqClient()

    cmd_vel = np.array([0.0, 0.0]) 
    result = env_client.step(cmd_vel, reset=True)

    print(f"Result: {result.keys()}")

    print(result["observation"])
    
    #print(f"Received Obs: {result['observation']} | Reward: {result['reward']}")
