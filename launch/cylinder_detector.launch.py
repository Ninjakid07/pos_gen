from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    # Start the cylinder detector node
    cylinder_detector_node = Node(
        package='pos_gen',
        executable='cylinder_detector',
        name='cylinder_detector',
        output='screen',
        parameters=[
            {
                # Point cloud processing parameters
                'voxel_leaf_size': 0.01,
                
                # Passthrough filter limits (in meters, relative to camera frame)
                'passthrough_x_min': -0.5,    # Left
                'passthrough_x_max': 0.5,     # Right
                'passthrough_y_min': -0.5,    # Down
                'passthrough_y_max': 0.5,     # Up
                'passthrough_z_min': 0.0,     # Minimum distance
                'passthrough_z_max': 0.7,     # Maximum distance (70cm)
                
                # Statistical outlier removal parameters
                'outlier_mean_k': 50,         # Number of nearest neighbors
                'outlier_std_dev': 1.0,       # Standard deviation multiplier
                
                # Cylinder segmentation parameters
                'cylinder_distance_threshold': 0.03,  # Max distance from point to model
                'cylinder_radius_min': 0.025,        # Minimum cylinder radius (2.5cm)
                'cylinder_radius_max': 0.035,        # Maximum cylinder radius (3.5cm)
                'normal_radius_search': 0.1,         # Radius for normal estimation
                'ransac_threshold': 0.02,            # RANSAC threshold
                'ransac_iterations': 10000,          # Maximum RANSAC iterations
                
                # Detection parameters
                'min_cylinder_points': 200,          # Minimum points for valid cylinder
                'min_confidence_threshold': 0.7,     # Minimum confidence score
                
                # Tracking parameters
                'tracking_distance_threshold': 0.1,  # Max distance for tracking (10cm)
                
                # Frame ID for publishing
                'frame_id': 'camera_depth_optical_frame',
                
                # Use simulation time
                'use_sim_time': use_sim_time
            }
        ],
        remappings=[
            # Adjust these remappings based on your actual topic names from OAK-D camera
            ('/oak/points', '/camera/depth/points'),
            ('/oak/rgb', '/camera/color/image_raw'),
            ('/oak/depth', '/camera/depth/image_rect_raw')
        ]
    )
    
    # Create launch description
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),
        cylinder_detector_node
    ])
