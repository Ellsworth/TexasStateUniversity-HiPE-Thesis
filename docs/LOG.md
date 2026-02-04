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

> TODO: Get the Cartographer 3D map visualized in RViz, it uses a custom message. Might want to figure out how to get RViz to load the plugin

```
root@hipe6-thrc:/workspace# ros2 topic info /submap_list
Type: cartographer_ros_msgs/msg/SubmapList
Publisher count: 1
Subscription count: 1
root@hipe6-thrc:/workspace# 
```

> The map seems to jitter/drift over time. Need to find out why.

# January 28

Got the windowed 'grid view' for the SLAM data working. Seems to work okay, but there's likely plenty of tuning that needs to be done here.

# February 3

The world control service is setup, along with a basic ZeroMQ bridge for the RL agent to 'teleop' the robot.

```ros2 service call /world/shapes/control ros_gz_interfaces/srv/ControlWorld \
"{world_control: {multi_step: 10, pause: true}}"```

We need to implement the state machine for Observe/Reward -> Act -> Run GZ for N ticks -> Repeat

# February 4

In order to reset the environment, we need to both restart cartographer and reset the position of the robot.

```python
import subprocess
import os
import signal

def restart_cartographer():
    # Find the process ID of the cartographer node
    # We use 'pgrep' to find the PID based on the executable name
    try:
        pid = subprocess.check_output(["pgrep", "-f", "cartographer_node"]).decode().strip()
        if pid:
            print(f"Restarting Cartographer (PID: {pid})...")
            # Send SIGTERM (or SIGKILL if it's being stubborn)
            os.kill(int(pid.split('\n')[0]), signal.SIGTERM)
    except subprocess.CalledProcessError:
        print("Cartographer node not found. Is it running?")
```
