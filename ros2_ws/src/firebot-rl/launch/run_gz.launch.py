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
    ros_gz_share = get_package_share_directory('ros_gz_sim')
    gz_sim_launch = PathJoinSubstitution(
        [ros_gz_share, 'launch', 'gz_sim.launch.py']
    )

    world = PathJoinSubstitution([
        FindPackageShare('firebot-rl'),
        'assets',
        'world-test.sdf'
    ])

    log = LogInfo(msg=["Resolved world path = ", world])


    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={
            'gz_args': [world, TextSubstitution(text=' -v 4')]
        }.items()
    )

    # -----------------------------
    # RViz2
    # -----------------------------
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        #arguments=[
        #    '-d',
        #    PathJoinSubstitution([
        #        get_package_share_directory('your_pkg'),
        #        'rviz',
        #        'your_config.rviz'
        #    ])
        #]
    )

    return LaunchDescription([log, gazebo, rviz])
