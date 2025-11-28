from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import TextSubstitution
from launch.actions import LogInfo

from launch_ros.actions import Node


def generate_launch_description():
    # -----------------------------
    # Gazebo Harmonic (ros_gz_sim)
    # -----------------------------
    ros_gz_share = get_package_share_directory("ros_gz_sim")
    gz_sim_launch = PathJoinSubstitution([ros_gz_share, "launch", "gz_sim.launch.py"])

    world = PathJoinSubstitution(
        [FindPackageShare("firebot-rl"), "assets", "world-test.sdf"]
    )

    log = LogInfo(msg=["Resolved world path = ", world])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={
            "gz_args": [world, TextSubstitution(text=" -v 1 -r")]
        }.items(),
    )

    robot_sdf = PathJoinSubstitution([
        FindPackageShare('firebot-rl'),
            'assets',
            'marble_hd2_sensor_config_4',
            '11',
            'model.sdf'
    ])

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-file",
            robot_sdf,
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

    # -----------------------------
    # RViz2
    # -----------------------------
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution([
                FindPackageShare("firebot-rl"),
                "rviz",
                "firebot_nav.rviz"
            ])
        ]
    )


    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="parameter_bridge",
        parameters=[{
            "bridge_names": ["front_laser"],

            "bridges.front_laser.ros_topic_name": "/marble_hd2/scan/front",
            "bridges.front_laser.gz_topic_name": "/world/shapes/model/marble_hd2/link/base_link/sensor/front_laser/scan",
            "bridges.front_laser.ros_type_name": "sensor_msgs/msg/LaserScan",
            "bridges.front_laser.gz_type_name": "gz.msgs.LaserScan",
            "bridges.front_laser.direction": "GZ_TO_ROS",
            "bridges.front_laser.lazy": False,
        }]
    )

    


    return LaunchDescription([log, gazebo, spawn_robot, rviz, bridge])
