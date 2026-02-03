import zmq
import numpy as np
import msgpack
import msgpack_numpy as m
import time
import matplotlib.pyplot as plt
import logging

m.patch()

# Configure logging with timestamp
logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# RLZmqClient moved to firebot_rl.rl_zmq_client
from firebot_rl.rl_zmq_client import RLZmqClient


class TeleopState:
    """
    WASD teleop:
      w/s -> +x / -x (linear)
      a/d -> +z / -z (angular)  (choose sign convention you want)
      space -> immediate stop
      r -> reset=True for one step
      q / esc -> quit
    """
    def __init__(self, v_lin=0.3, v_ang=1.0):
        self.pressed = set()
        self.v_lin = float(v_lin)
        self.v_ang = float(v_ang)
        self.reset_one_shot = False
        self.quit = False

    def on_key_press(self, event):
        k = (event.key or "").lower()
        if k in ("q", "escape"):
            self.quit = True
            return
        if k == "r":
            self.reset_one_shot = True
            return
        if k == " " or k == "space":
            # treat as pressed for the next compute, then clear in compute
            self.pressed.add("space")
            return
        self.pressed.add(k)

    def on_key_release(self, event):
        k = (event.key or "").lower()
        self.pressed.discard(k)
        if k == " " or k == "space":
            self.pressed.discard("space")

    def compute_cmd_vel(self) -> np.ndarray:
        lin = 0.0
        ang = 0.0

        if "space" in self.pressed:
            # immediate stop
            self.pressed.discard("space")
            return np.array([0.0, 0.0], dtype=np.float32)

        if "w" in self.pressed:
            lin += self.v_lin
        if "s" in self.pressed:
            lin -= self.v_lin

        # NOTE: pick your angular sign convention here.
        if "a" in self.pressed:
            ang += self.v_ang
        if "d" in self.pressed:
            ang -= self.v_ang

        return np.array([lin, ang], dtype=np.float32)

    def pop_reset(self) -> bool:
        r = self.reset_one_shot
        self.reset_one_shot = False
        return r


def main():
    env_client = RLZmqClient(ip="127.0.0.1", port=5555)
    teleop = TeleopState(v_lin=1.0, v_ang=0.75)

    # --- Matplotlib live view setup ---
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("Live Observation + WASD Teleop")

    # Start with a placeholder until first observation arrives
    obs = np.zeros((64, 64), dtype=np.int16).T
    im = ax.imshow(obs)
    ax.set_title("Observation (int16). Focus this window for WASD.")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Key events (Matplotlib only receives keys when this window has focus)
    fig.canvas.mpl_connect("key_press_event", teleop.on_key_press)
    fig.canvas.mpl_connect("key_release_event", teleop.on_key_release)

    dt = 0.05  # 20 Hz update/send rate

    # If you prefer to keep sending even when window not focused,
    # you'll need OS-level key capture (e.g., pynput), not MPL events.
    while not teleop.quit and plt.fignum_exists(fig.number):
        cmd_vel = teleop.compute_cmd_vel()
        reset = teleop.pop_reset()

        try:
            result = env_client.step(cmd_vel, reset=reset)
        except KeyboardInterrupt:
            break
        
        print(f"Received result: {result}")

        # Expect result["observation"] to already be a numpy array via msgpack_numpy
        obs = result.get("observation", None)
        if obs is not None:
            obs = np.asarray(obs)

            # Ensure int16 as stated
            if obs.dtype != np.int16:
                obs = obs.astype(np.int16, copy=False)

            if obs.size > 0:
                # If observation can change shape, handle it
                if im.get_array().shape != obs.shape:
                    im.set_data(obs)
                    ax.set_xlim(-0.5, obs.shape[1] - 0.5)
                    ax.set_ylim(obs.shape[0] - 0.5, -0.5)
                else:
                    im.set_data(obs)

                # Dynamic scaling per frame
                im.autoscale()

        fig.canvas.draw_idle()
        plt.pause(0.001)
        time.sleep(dt)

    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    main()
