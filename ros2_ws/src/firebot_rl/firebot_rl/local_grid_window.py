#!/usr/bin/env python3
"""
ROS 2 node: subscribes to nav_msgs/OccupancyGrid and publishes an MxN window
centered on the robot, rotated so the window is aligned with robot heading.

- Window is in robot frame orientation: "forward" is +X of base_frame.
- Output is an Int16MultiArray shaped (M rows, N cols).
- pad_value defaults to -1 (unknown).
- Rotation uses nearest-neighbor sampling (fast, discrete-safe for occupancy).

Topics:
  Sub:  /map (nav_msgs/OccupancyGrid)            [param: map_topic]
  Pub:  /local_grid (std_msgs/Int16MultiArray)  [param: out_topic]

Frames:
  base_frame (default: base_link)
  map_frame  (default: map)
"""

import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Int16MultiArray, MultiArrayDimension

import tf2_ros
from geometry_msgs.msg import TransformStamped


def quat_to_yaw(q) -> float:
    # yaw (z-rotation) from quaternion
    # q has fields x,y,z,w
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class LocalGridWindowNode(Node):
    def __init__(self):
        super().__init__("local_grid_window")

        # Parameters
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("out_topic", "/local_grid")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("N", 65)          # cols
        self.declare_parameter("M", 65)          # rows
        self.declare_parameter("pad_value", -1)
        self.declare_parameter("publish_hz", 5.0)

        self.map_topic = self.get_parameter("map_topic").value
        self.out_topic = self.get_parameter("out_topic").value
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.N = int(self.get_parameter("N").value)
        self.M = int(self.get_parameter("M").value)
        self.pad_value = int(self.get_parameter("pad_value").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)

        if self.N <= 0 or self.M <= 0:
            raise ValueError("N and M must be positive.")

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.grid_msg: Optional[OccupancyGrid] = None

        # ROS I/O
        self.sub = self.create_subscription(OccupancyGrid, self.map_topic, self.on_grid, 1)
        self.pub = self.create_publisher(Int16MultiArray, self.out_topic, 1)

        period = 1.0 / max(self.publish_hz, 0.1)
        self.timer = self.create_timer(period, self.on_timer)

        self.get_logger().info(
            f"Listening {self.map_topic}, publishing {self.out_topic} ({self.M}x{self.N}), "
            f"frames: {self.map_frame} <- {self.base_frame}"
        )

    def on_grid(self, msg: OccupancyGrid):
        self.grid_msg = msg

    def lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        """
        Returns (x, y, yaw) of base_frame in map_frame.
        """
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except Exception:
            return None

        x = float(t.transform.translation.x)
        y = float(t.transform.translation.y)
        yaw = quat_to_yaw(t.transform.rotation)
        return x, y, yaw

    @staticmethod
    def world_to_grid(grid: OccupancyGrid, x_m: float, y_m: float) -> Tuple[int, int]:
        res = float(grid.info.resolution)
        ox = float(grid.info.origin.position.x)
        oy = float(grid.info.origin.position.y)
        gx = int(math.floor((x_m - ox) / res))
        gy = int(math.floor((y_m - oy) / res))
        return gx, gy

    @staticmethod
    def extract_window_no_rotate(
        data2d: np.ndarray,
        gx: int,
        gy: int,
        N: int,
        M: int,
        pad_value: int,
    ) -> np.ndarray:
        """
        data2d: shape (H, W) indexed as [y, x]
        Returns window (M, N) with padding.
        """
        H, W = data2d.shape
        half_w = N // 2
        half_h = M // 2

        x0 = gx - half_w
        y0 = gy - half_h
        x1 = x0 + N - 1
        y1 = y0 + M - 1

        out = np.full((M, N), pad_value, dtype=np.int16)

        sx0 = max(x0, 0)
        sy0 = max(y0, 0)
        sx1 = min(x1, W - 1)
        sy1 = min(y1, H - 1)

        if sx0 <= sx1 and sy0 <= sy1:
            dx0 = sx0 - x0
            dy0 = sy0 - y0
            dx1 = dx0 + (sx1 - sx0)
            dy1 = dy0 + (sy1 - sy0)

            out[dy0 : dy1 + 1, dx0 : dx1 + 1] = data2d[sy0 : sy1 + 1, sx0 : sx1 + 1]

        return out

    @staticmethod
    def rotate_about_center_nearest(img: np.ndarray, yaw_rad: float, pad_value: int) -> np.ndarray:
        """
        Rotate img by angle (radians) about its center using nearest-neighbor sampling.
        Positive yaw rotates counter-clockwise.

        For "aligned to robot heading", we want to rotate the map patch by -yaw,
        so that robot forward points "up"/+x in the patch coordinates consistently.
        """
        H, W = img.shape
        cy = (H - 1) * 0.5
        cx = (W - 1) * 0.5

        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)

        # Destination grid coordinates
        ys, xs = np.indices((H, W), dtype=np.float32)
        x = xs - cx
        y = ys - cy

        # Inverse map: src = R^-1 * dst. For rotation by theta, inverse is rotation by -theta.
        # Here we'll interpret yaw_rad as the desired rotation applied to the image.
        # To compute each dst pixel from src, use inverse rotation:
        # [sx] =  c  s [x]
        # [sy] = -s  c [y]
        sx = (c * x + s * y) + cx
        sy = (-s * x + c * y) + cy

        sxn = np.rint(sx).astype(np.int32)
        syn = np.rint(sy).astype(np.int32)

        out = np.full((H, W), pad_value, dtype=np.int16)
        mask = (sxn >= 0) & (sxn < W) & (syn >= 0) & (syn < H)
        out[mask] = img[syn[mask], sxn[mask]]
        return out

    def aligned_window(
        self,
        grid: OccupancyGrid,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        N: int,
        M: int,
        pad_value: int,
    ) -> np.ndarray:
        """
        Steps:
          1) Extract a big square patch around robot in map grid coords.
             side = ceil(sqrt(N^2 + M^2)) (so rotation won't clip), forced odd.
          2) Rotate patch by -robot_yaw.
          3) Crop center MxN.
        """
        W = int(grid.info.width)
        H = int(grid.info.height)

        gx, gy = self.world_to_grid(grid, robot_x, robot_y)

        data = np.asarray(grid.data, dtype=np.int16).reshape((H, W))

        # Big square side to avoid clipping after rotation
        side = int(math.ceil(math.sqrt(N * N + M * M)))
        if side % 2 == 0:
            side += 1

        big = self.extract_window_no_rotate(data, gx, gy, side, side, pad_value)

        # Rotate by -yaw to align to robot heading
        rot = self.rotate_about_center_nearest(big, -robot_yaw, pad_value)

        # Crop center
        cy = side // 2
        cx = side // 2
        y0 = cy - (M // 2)
        x0 = cx - (N // 2)
        out = rot[y0 : y0 + M, x0 : x0 + N]

        # If M or N even, the above slicing still yields (M,N) correctly.
        return out

    def on_timer(self):
        if self.grid_msg is None:
            return

        pose = self.lookup_robot_pose()
        if pose is None:
            self.get_logger().warn("TF lookup failed (map <- base). Not publishing.")
            return

        x, y, yaw = pose

        window = self.aligned_window(
            self.grid_msg, x, y, yaw, self.N, self.M, self.pad_value
        )  # (M, N), int16

        msg = Int16MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="rows", size=self.M, stride=self.M * self.N),
            MultiArrayDimension(label="cols", size=self.N, stride=self.N),
        ]
        msg.data = window.reshape(-1).astype(np.int16).tolist()

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = LocalGridWindowNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
