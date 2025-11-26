FROM ros:jazzy-ros-base-noble

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz
