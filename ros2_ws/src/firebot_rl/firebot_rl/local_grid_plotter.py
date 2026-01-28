#!/usr/bin/env python3
"""
Subscribe to std_msgs/Int16MultiArray (MxN local grid) and display as a live-updating plot
with an arrow at the center representing the robot.

Assumptions:
- Incoming grid is already rotated/aligned to robot heading.
- Arrow points "forward" in the local grid frame. By convention here: +X (to the right).
  Change arrow_dir to "up" if your convention is forward=+Y.

Run:
  ros2 run <your_pkg> local_grid_plot
  or python3 local_grid_plot.py --ros-args -p topic:=/local_grid -p refresh_hz:=20.0
"""

import threading
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

import matplotlib.pyplot as plt


class LocalGridPlotter(Node):
    def __init__(self):
        super().__init__("local_grid_plotter")

        self.declare_parameter("topic", "/local_grid")
        self.declare_parameter("refresh_hz", 20.0)
        self.declare_parameter("arrow_length_cells", 6.0)
        self.declare_parameter("arrow_dir", "right")  # "right" or "up"

        self.topic = self.get_parameter("topic").value
        self.refresh_hz = float(self.get_parameter("refresh_hz").value)
        self.arrow_len = float(self.get_parameter("arrow_length_cells").value)
        self.arrow_dir = str(self.get_parameter("arrow_dir").value).lower()

        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_shape: Optional[Tuple[int, int]] = None

        self.sub = self.create_subscription(Int16MultiArray, self.topic, self._on_msg, 10)

        # Matplotlib setup
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.robot_arrow = None

        self.ax.set_title(f"Local Grid: {self.topic}")
        self.ax.set_xlabel("x (cols)")
        self.ax.set_ylabel("y (rows)")

        period = 1.0 / max(self.refresh_hz, 1.0)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"Subscribed to {self.topic}, refreshing plot at {self.refresh_hz} Hz"
        )

    def _on_msg(self, msg: Int16MultiArray):
        rows = cols = None
        if msg.layout.dim and len(msg.layout.dim) >= 2:
            rows = int(msg.layout.dim[0].size)
            cols = int(msg.layout.dim[1].size)

        data = np.asarray(msg.data, dtype=np.int16)

        if rows is not None and cols is not None and rows * cols == data.size:
            grid = data.reshape((rows, cols))
        else:
            side = int(np.sqrt(data.size))
            if side * side == data.size:
                grid = data.reshape((side, side))
            else:
                self.get_logger().warn(
                    f"Cannot reshape data of size {data.size}. "
                    f"layout rows={rows} cols={cols}. Skipping."
                )
                return

        with self._lock:
            self._latest = grid
            self._latest_shape = grid.shape

    def _ensure_robot_arrow(self, rows: int, cols: int):
        """
        Draw (or redraw) the arrow at the center in data coordinates.
        For imshow with origin='lower', cell centers are at integer coordinates.
        """
        cx = (cols - 1) / 2.0
        cy = (rows - 1) / 2.0

        if self.arrow_dir == "up":
            dx, dy = 0.0, self.arrow_len
        else:  # default "right"
            dx, dy = self.arrow_len, 0.0

        # Remove old arrow if present (e.g., on resize/shape change)
        if self.robot_arrow is not None:
            try:
                self.robot_arrow.remove()
            except Exception:
                pass
            self.robot_arrow = None

        # Use annotate arrow for simple updating/redrawing
        self.robot_arrow = self.ax.annotate(
            "", xy=(cx + dx, cy + dy), xytext=(cx, cy),
            arrowprops=dict(arrowstyle="->", linewidth=2, color="red"),
            zorder=5,
        )

    def _on_timer(self):
        with self._lock:
            grid = None if self._latest is None else self._latest.copy()

        if grid is None:
            plt.pause(0.001)
            return

        rows, cols = grid.shape

        if self.im is None:
            self.im = self.ax.imshow(grid, origin="lower", interpolation="nearest")
            self.fig.colorbar(self.im, ax=self.ax)
            self._ensure_robot_arrow(rows, cols)
        else:
            # If shape changed, recreate axes content
            if self.im.get_array().shape != grid.shape:
                self.ax.cla()
                self.ax.set_title(f"Local Grid: {self.topic}")
                self.ax.set_xlabel("x (cols)")
                self.ax.set_ylabel("y (rows)")
                self.im = self.ax.imshow(grid, origin="lower", interpolation="nearest")
                self.fig.colorbar(self.im, ax=self.ax)
                self._ensure_robot_arrow(rows, cols)
            else:
                self.im.set_data(grid)
                # Arrow stays centered if shape unchanged; nothing to update.

        # Keep a stable color scale (occupancy conventions)
        self.im.set_clim(vmin=-1, vmax=100)

        self.fig.canvas.draw_idle()
        plt.pause(0.001)


def main():
    rclpy.init()
    node = LocalGridPlotter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
