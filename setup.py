from setuptools import find_packages, setup

package_name = 'pos_gen'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Removed launch file reference since it doesn't exist yet
    ],
    install_requires=[
        'setuptools',
        'rclpy',  # Changed from rospy to rclpy for ROS2
        'numpy',  # Added numpy dependency
    ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Tree position marker generator and navigation system for ROSbot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tree_marker_publisher = pos_gen.tree_marker_publisher:main',
            'row_nav = pos_gen.row_nav:main',
            'tree_row_navigator = pos_gen.tree_row_navigator:main',
            'cylinder_analyzer = pos_gen.cylinder_analyzer:main',
        ],
    },
)
