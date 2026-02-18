FROM docker.io/library/ros:jazzy-ros-base-noble

RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-cartographer \
    ros-jazzy-cartographer-ros \
    ros-jazzy-image-view \
    ros-jazzy-web-video-server