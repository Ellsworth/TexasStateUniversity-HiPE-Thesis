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

Cartographer works, but the provided 2D occupancy grid leaves some to be desired. Things like ledges aren't depicted well in the map.

> Next steps: Get the Cartographer 3D map visualized in RViz, it uses a custom message. Might want to figure out how to get RViz to load the plugin

```
root@hipe6-thrc:/workspace# ros2 topic info /submap_list
Type: cartographer_ros_msgs/msg/SubmapList
Publisher count: 1
Subscription count: 1
root@hipe6-thrc:/workspace# 
```