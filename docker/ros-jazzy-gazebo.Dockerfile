FROM docker.io/library/ros:jazzy-ros-base-noble

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-cartographer \
    ros-jazzy-cartographer-ros \
    ros-jazzy-image-view \
    ros-jazzy-web-video-server \
    && rm -rf /var/lib/apt/lists/*

# Copy source files to install dependencies
COPY ros2_ws/src /tmp/ros2_ws/src
WORKDIR /tmp/ros2_ws

# Install dependencies
RUN . /opt/ros/jazzy/setup.sh && \
    apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*