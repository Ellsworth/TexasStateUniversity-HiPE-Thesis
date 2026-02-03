import zmq
import msgpack
import msgpack_numpy as m
import logging

m.patch()

class RLZmqClient:
    def __init__(self, ip="127.0.0.1", port=5555, timeout_ms=15000):
        self.ip = ip
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context.instance()
        self.socket = None
        self._connect()

    def _connect(self):
        """Connects or Reconnects to the server"""
        if self.socket:
            self.socket.close()
        
        self.socket = self.context.socket(zmq.REQ)
        # Set timeouts
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0) # Don't hang on close
        
        self.socket.connect(f"tcp://{self.ip}:{self.port}")
        logging.info(f"Connected to ZMQ server at {self.ip}:{self.port}")

    def step(self, cmd_vel, steps: int = 100, reset: bool = False):
        payload = {
            "command": "step",
            "cmd_vel": cmd_vel,   # np array, e.g. shape (2,)
            "step": steps,
            "reset": reset
        }

        try:
            # Use msgpack options that behave well with numpy + bytes
            logging.info(f"Sending payload: cmd_vel={cmd_vel}, step={payload['step']}")
            self.socket.send(msgpack.packb(payload, use_bin_type=True))
            
            message = self.socket.recv()
            response = msgpack.unpackb(message, raw=False)
            logging.info(f"Received response: {response.keys() if isinstance(response, dict) else response}")
            return response
        except zmq.Again:
            logging.warning(f"Request timed out ({self.timeout_ms}ms). Reconnecting...")
            self._connect()
            return {} # Return empty dict to indicate failure/timeout without crashing
        except zmq.ZMQError as e:
            logging.error(f"ZMQ Error: {e}. Reconnecting...")
            self._connect()
            return {}
