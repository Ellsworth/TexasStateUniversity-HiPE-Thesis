from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'firebot_rl'

# Recursively collect all files inside assets/
asset_files = [
    (os.path.join('share', package_name, root.replace('assets/', 'assets/')),
     [os.path.join(root, f) for f in files])
    for root, dirs, files in os.walk('assets')
    if files
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.*')), # Launch files
        ('share/' + package_name + '/config', glob('config/*')), # Config files
    ] + asset_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
                'grid_window_publisher = firebot_rl.local_grid_window:main',
                'grid_window_plotter = firebot_rl.local_grid_plotter:main',
                'zmq_bridge = firebot_rl.zmq_bridge:main',
                'contact_monitor = firebot_rl.contact_monitor:main',
        ],
    },
)
