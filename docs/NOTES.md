# Notes

## January 20
* Basic setup, get repo updated, setup machine.
* Converted from 2D LiDAR -> 3D LiDAR. Displays in rviz

### Next Steps
* Get cartographer working, we are going to need 3D LiDAR likely.

## January 21
* 3D LiDAR is working
* Divert to getting the HD2 moving

## January 22

```bash
gz topic -t /model/marble_hd2/link/left_track/track_cmd_vel  -m gz.msgs.Double -p 'data: 1.0'
gz topic -t /model/marble_hd2/link/right_track/track_cmd_vel -m gz.msgs.Double -p 'data: 1.0'
```

This worked! ```/cmd_vel``` works as well. 

# January 27

Cartographer runs with the following command.

```/opt/ros/jazzy/lib/cartographer_ros/cartographer_node -configuration_directory /workspace/ros2_ws/install/firebot-rl/share/firebot-rl/config -configuration_basename cartographer.lua```