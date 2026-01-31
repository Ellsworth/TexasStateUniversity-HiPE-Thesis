default:
  podman compose down
  gnome-terminal \
    --tab -- bash -c "podman compose up"

amd:
  nix-shell -p xorg.xhost --run "xhost +SI:localuser:$USER"
  podman compose down
  podman compose -f compose-amd.yml up

shell:
  podman exec -it gazebo bash -c "source /workspace/ros2_ws/install/setup.bash; exec /bin/bash"

clean:
  rm -r ros2_ws/build/ ros2_ws/install/ ros2_ws/log/

stop:
  podman compose down
