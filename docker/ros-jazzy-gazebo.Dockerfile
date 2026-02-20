FROM docker.io/library/ros:jazzy-ros-base-noble

RUN apt-get update && apt-get install -y \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Copy source files to install dependencies
COPY ros2_ws/src /tmp/ros2_ws/src
WORKDIR /tmp/ros2_ws

# Install dependencies
RUN . /opt/ros/jazzy/setup.sh && \
    apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    rm -rf /var/lib/apt/lists/*