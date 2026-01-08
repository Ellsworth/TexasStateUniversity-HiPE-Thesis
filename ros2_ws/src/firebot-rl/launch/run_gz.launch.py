import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def generate_launch_description():
    # -----------------------------
    # Paths & Setup
    # -----------------------------
    pkg_share = get_package_share_directory("firebot-rl")

    ros_gz_share = get_package_share_directory("ros_gz_sim")
    gz_sim_launch = PathJoinSubstitution([ros_gz_share, "launch", "gz_sim.launch.py"])

    world = PathJoinSubstitution([pkg_share, "assets", "world-test.sdf"])

    # We use os.path.join here so we can open the file immediately for the publisher
    robot_sdf_path = os.path.join(
        pkg_share, "assets", "marble_hd2_sensor_config_4", "11", "model.sdf"
    )

    # -----------------------------
    # Nodes & Actions
    # -----------------------------

    log_world_path = LogInfo(msg=["Resolved world path = ", world])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={
            "gz_args": [world, TextSubstitution(text=" -v 1 -r")]
        }.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-file",
            robot_sdf_path,
            "-name",
            "marble_hd2",
            "-x",
            "5",
            "-y",
            "30",
            "-z",
            "7.5",
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {
                "use_sim_time": True,
                # Read the file content directly using the absolute path
                "robot_description": open(robot_sdf_path).read(),
            }
        ],
    )

    bridge_config = PathJoinSubstitution([pkg_share, "config", "bridge_config.yaml"])

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="parameter_bridge",
        parameters=[{"config_file": bridge_config}],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", os.path.join(pkg_share, "rviz", "firebot_nav.rviz")],
    )

    log_bridge = LogInfo(msg=["Using Bridge Config: ", bridge_config])

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "-configuration_directory",
            os.path.join(pkg_share, "config"),
            "-configuration_basename",
            "cartographer.lua",
        ],
    )
    
    """"
    cartographer_occupancy_grid_node = Node(
            package="cartographer_ros",
            executable="occupancy_grid_node",
            name="occupancy_grid_node",
            output="screen",
            parameters=[{"use_sim_time": True}],
            arguments=["-resolution", "0.05", "-publish_period_sec", "1.0"],
    )
    """

    return LaunchDescription(
        [
            log_world_path,
            log_bridge,
            gazebo,
            spawn_robot,
            robot_state_publisher,
            bridge,
            rviz,
            #cartographer_node,
            #cartographer_occupancy_grid_node,
        ]
    )
