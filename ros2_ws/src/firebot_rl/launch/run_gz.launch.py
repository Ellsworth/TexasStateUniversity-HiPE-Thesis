import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch.launch_description_sources import AnyLaunchDescriptionSource

def _read_robot_description(context, robot_sdf_path_lc_name: str):
    """
    Read robot SDF content at launch runtime (not at file-parse time) so failures
    are easier to diagnose and paths can be overridden via launch args.
    """
    robot_sdf_path = LaunchConfiguration(robot_sdf_path_lc_name).perform(context)

    if not os.path.isabs(robot_sdf_path):
        # Allow relative paths (relative to cwd), but prefer absolute.
        robot_sdf_path = os.path.abspath(robot_sdf_path)

    if not os.path.exists(robot_sdf_path):
        raise RuntimeError(f"Robot SDF does not exist: {robot_sdf_path}")

    with open(robot_sdf_path, "r", encoding="utf-8") as f:
        robot_description = f.read()

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
    )

    return [LogInfo(msg=["robot_state_publisher using: ", robot_sdf_path]), rsp]


def generate_launch_description():
    pkg_share = get_package_share_directory("firebot_rl")
    ros_gz_share = get_package_share_directory("ros_gz_sim")

    # -----------------------------
    # Launch arguments
    # -----------------------------
    world_arg = DeclareLaunchArgument(
        "world",
        default_value=PathJoinSubstitution([pkg_share, "assets", "world-test.sdf"]),
        description="Path to the Gazebo world SDF",
    )

    robot_sdf_arg = DeclareLaunchArgument(
        "robot_sdf",
        default_value=TextSubstitution(
            text=os.path.join(
                pkg_share, "assets", "marble_hd2_sensor_config_4", "11", "model.sdf"
            )
        ),
        description="Path to robot SDF to spawn and publish",
    )

    robot_name_arg = DeclareLaunchArgument(
        "robot_name",
        default_value=TextSubstitution(text="marble_hd2"),
        description="Spawned robot entity name in Gazebo",
    )

    # Pose arguments (strings are fine; ros_gz_sim/create expects strings)
    x_arg = DeclareLaunchArgument("x", default_value=TextSubstitution(text="5"))
    y_arg = DeclareLaunchArgument("y", default_value=TextSubstitution(text="30"))
    z_arg = DeclareLaunchArgument("z", default_value=TextSubstitution(text="8"))

    gz_verbosity_arg = DeclareLaunchArgument(
        "gz_verbosity",
        default_value=TextSubstitution(text="1"),
        description="Gazebo verbosity level (-v)",
    )



    use_rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value=TextSubstitution(text="true"),
        description="Launch RViz2",
    )

    rviz_config_arg = DeclareLaunchArgument(
        "rviz_config",
        default_value=TextSubstitution(text=os.path.join(pkg_share, "config", "firebot.rviz")),
        description="RViz config file",
    )

    use_cartographer_arg = DeclareLaunchArgument(
        "cartographer",
        default_value=TextSubstitution(text="true"),
        description="Launch Cartographer",
    )

    carto_config_dir_arg = DeclareLaunchArgument(
        "carto_config_dir",
        default_value=TextSubstitution(text=os.path.join(pkg_share, "config")),
        description="Cartographer configuration directory",
    )

    carto_basename_arg = DeclareLaunchArgument(
        "carto_basename",
        default_value=TextSubstitution(text="cartographer.lua"),
        description="Cartographer configuration basename",
    )

    # -----------------------------
    # Gazebo include
    # -----------------------------
    gz_sim_launch = PathJoinSubstitution([ros_gz_share, "launch", "gz_sim.launch.py"])

    # gz_args is a single string in many ros_gz_sim versions; pass as one composed string
    # Default to headless (-s). Set GZ_GUI=1 to launch with the GUI.
    server_only = "" if os.environ.get("GZ_GUI", "0") == "1" else " -s"
    gz_args = [
        LaunchConfiguration("world"),
        TextSubstitution(text=f" -r{server_only} -v "),
        LaunchConfiguration("gz_verbosity"),
    ]

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gz_sim_launch),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    # -----------------------------
    # Read Gazebo-ROS bridge launchfile
    # -----------------------------
    bridge_xml = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([pkg_share, "launch", "bridge.launch.xml"])
        ),
        launch_arguments={}.items(),
    )

    # -----------------------------
    # Spawn robot
    # -----------------------------
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-file",
            LaunchConfiguration("robot_sdf"),
            "-name",
            LaunchConfiguration("robot_name"),
            "-x",
            LaunchConfiguration("x"),
            "-y",
            LaunchConfiguration("y"),
            "-z",
            LaunchConfiguration("z"),
        ],
    )

    # -----------------------------
    # RViz
    # -----------------------------
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # -----------------------------
    # Cartographer
    # -----------------------------

    cartographer_node = Node(
        package="cartographer_ros",
        executable="cartographer_node",
        name="cartographer_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "-configuration_directory", LaunchConfiguration("carto_config_dir"),
            "-configuration_basename", LaunchConfiguration("carto_basename"),
        ],
        remappings=[
            ("points2", "/marble_hd2/scan"),
            ("imu", "/marble_hd2/imu"),
        ],
        condition=IfCondition(LaunchConfiguration("cartographer")),
        respawn=True,
        respawn_delay=2.0,
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        # -resolution: size of map pixels in meters
        # -publish_period_sec: how often to refresh the map in RViz
        arguments=['-resolution', '0.25', '-publish_period_sec', '0.1']
    )

    # -----------------------------
    # Logging
    # -----------------------------
    log_settings = LogInfo(
        msg=[
            "world=",
            LaunchConfiguration("world"),
            " robot_sdf=",
            LaunchConfiguration("robot_sdf"),
            " name=",
            LaunchConfiguration("robot_name"),
            " pose=(",
            LaunchConfiguration("x"),
            ", ",
            LaunchConfiguration("y"),
            ", ",
            LaunchConfiguration("z"),
        ]
    )

    # robot_state_publisher created via OpaqueFunction so we can read the file at runtime
    rsp_runtime = OpaqueFunction(function=lambda ctx: _read_robot_description(ctx, "robot_sdf"))

    grid_window_plotter = Node(
        package='firebot_rl',
        executable='grid_window_plotter',
        name='grid_window_plotter',
        output='screen',
        parameters=[{'refresh_hz': 10.0}],
        arguments=[]
    )

    grid_window_publisher = Node(
        package='firebot_rl',
        executable='grid_window_publisher',
        name='grid_window_publisher',
        output='screen',
        parameters=[{'publish_hz': 10.0}],
        arguments=[]
    )

    zmq_bridge = Node(
        package='firebot_rl',
        executable='zmq_bridge',
        name='zmq_bridge',
        output='screen',
        parameters=[],
        arguments=[]
    )

    contact_monitor = Node(
        package='firebot_rl',
        executable='contact_monitor',
        name='contact_monitor',
        output='screen',
        parameters=[],
        arguments=[]
    )

    return LaunchDescription(
        [
            world_arg,
            robot_sdf_arg,
            robot_name_arg,
            x_arg,
            y_arg,
            z_arg,
            gz_verbosity_arg,

            use_rviz_arg,
            rviz_config_arg,
            use_cartographer_arg,
            carto_config_dir_arg,
            LogInfo(msg=['Cartographer config directory is: ', LaunchConfiguration('carto_config_dir')]),
            carto_basename_arg,
            log_settings,
            gazebo,
            bridge_xml,
            spawn_robot,
            rsp_runtime,
            rviz,
            cartographer_node,
            occupancy_grid_node,
            grid_window_plotter,
            grid_window_publisher,
            zmq_bridge,
            contact_monitor,
        ]
    )
