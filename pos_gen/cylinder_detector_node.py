#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError

import pyransac3d as pyrsc
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header, ColorRGBA, Int32

import time
import math
import struct
import ctypes
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

# PCL Point Cloud processing
import open3d as o3d
import open3d.visualization.rendering as rendering


@dataclass
class CylinderDetection:
    """Data class for cylinder detection results"""
    center: np.ndarray  # 3D position of cylinder center [x, y, z]
    axis: np.ndarray    # 3D orientation of cylinder axis [x, y, z]
    radius: float       # Radius of cylinder in meters
    confidence: float   # Confidence score (0-1)
    point_count: int    # Number of points supporting this cylinder
    tracking_id: int = -1  # Tracking ID (-1 if not tracked yet)
    age: int = 0        # Number of consecutive frames this cylinder has been detected


class CylinderDetector(Node):
    """ROS 2 node for detecting cylinders in point clouds"""
    
    def __init__(self):
        super().__init__('cylinder_detector')

        # Create QoS profile for better point cloud handling
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # Point cloud processing parameters
                ('voxel_leaf_size', 0.01),
                
                # Passthrough filter limits
                ('passthrough_x_min', -0.5),
                ('passthrough_x_max', 0.5),
                ('passthrough_y_min', -0.5),
                ('passthrough_y_max', 0.5),
                ('passthrough_z_min', 0.0),
                ('passthrough_z_max', 0.7),
                
                # Statistical outlier removal parameters
                ('outlier_mean_k', 50),
                ('outlier_std_dev', 1.0),
                
                # Cylinder segmentation parameters
                ('cylinder_distance_threshold', 0.03),
                ('cylinder_radius_min', 0.025),
                ('cylinder_radius_max', 0.035),
                ('normal_radius_search', 0.1),
                ('ransac_threshold', 0.02),
                ('ransac_iterations', 10000),
                
                # Detection parameters
                ('min_cylinder_points', 200),
                ('min_confidence_threshold', 0.7),
                
                # Tracking parameters
                ('tracking_distance_threshold', 0.1),
                
                # Frame ID for publishing
                ('frame_id', 'camera_depth_optical_frame'),
            ]
        )

        # Get parameters
        self.voxel_leaf_size = self.get_parameter('voxel_leaf_size').value
        
        self.passthrough_x_min = self.get_parameter('passthrough_x_min').value
        self.passthrough_x_max = self.get_parameter('passthrough_x_max').value
        self.passthrough_y_min = self.get_parameter('passthrough_y_min').value
        self.passthrough_y_max = self.get_parameter('passthrough_y_max').value
        self.passthrough_z_min = self.get_parameter('passthrough_z_min').value
        self.passthrough_z_max = self.get_parameter('passthrough_z_max').value
        
        self.outlier_mean_k = self.get_parameter('outlier_mean_k').value
        self.outlier_std_dev = self.get_parameter('outlier_std_dev').value
        
        self.cylinder_distance_threshold = self.get_parameter('cylinder_distance_threshold').value
        self.cylinder_radius_min = self.get_parameter('cylinder_radius_min').value
        self.cylinder_radius_max = self.get_parameter('cylinder_radius_max').value
        self.normal_radius_search = self.get_parameter('normal_radius_search').value
        self.ransac_threshold = self.get_parameter('ransac_threshold').value
        self.ransac_iterations = self.get_parameter('ransac_iterations').value
        
        self.min_cylinder_points = self.get_parameter('min_cylinder_points').value
        self.min_confidence_threshold = self.get_parameter('min_confidence_threshold').value
        
        self.tracking_distance_threshold = self.get_parameter('tracking_distance_threshold').value
        
        self.frame_id = self.get_parameter('frame_id').value
        
        # Initialize tracking
        self.previous_cylinders = []
        self.next_tracking_id = 0
        
        # Create CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize latest images
        self.latest_depth_image = None
        self.latest_rgb_image = None
        
        # Create subscribers
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/oak/points',
            self.point_cloud_callback,
            qos_profile
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/oak/depth',
            self.depth_image_callback,
            qos_profile
        )
        
        self.rgb_sub = self.create_subscription(
            Image,
            '/oak/rgb',
            self.rgb_image_callback,
            qos_profile
        )
        
        # Create publishers
        self.filtered_cloud_pub = self.create_publisher(
            PointCloud2,
            'filtered_cloud',
            10
        )
        
        self.cylinder_poses_pub = self.create_publisher(
            PoseArray,
            'cylinder_poses',
            10
        )
        
        self.marker_pub = self.create_publisher(
            MarkerArray,
            'cylinder_markers',
            10
        )
        
        self.cylinder_count_pub = self.create_publisher(
            Int32,
            'cylinder_count',
            10
        )
        
        self.get_logger().info('Cylinder detector initialized with parameters:')
        self.get_logger().info(f'  Cylinder radius range: {self.cylinder_radius_min:.3f} - {self.cylinder_radius_max:.3f} m')
        self.get_logger().info(f'  Detection range: {self.passthrough_z_max:.1f} m')
        self.get_logger().info(f'  Confidence threshold: {self.min_confidence_threshold:.2f}')
    
    def point_cloud_callback(self, msg):
        """Process incoming point cloud messages"""
        self.get_logger().debug(f'Received point cloud with {msg.width * msg.height} points')
        
        # Convert ROS PointCloud2 to Open3D point cloud
        start_time = time.time()
        o3d_cloud = self.ros_to_o3d(msg)
        
        if o3d_cloud is None or len(o3d_cloud.points) < self.min_cylinder_points:
            self.get_logger().warn('Received empty or too small point cloud')
            return
        
        # Process the point cloud
        self.process_point_cloud(o3d_cloud, msg.header)
        self.get_logger().debug(f'Processing time: {time.time() - start_time:.3f} seconds')
    
    def ros_to_o3d(self, ros_cloud):
        """Convert ROS PointCloud2 to Open3D point cloud"""
        # Extract points from ROS message
        field_names = [field.name for field in ros_cloud.fields]
        
        # Check if required fields are present
        if 'x' not in field_names or 'y' not in field_names or 'z' not in field_names:
            self.get_logger().error('Point cloud does not contain XYZ data')
            return None
        
        # Create numpy structured array to store points
        cloud_data = np.zeros(ros_cloud.width * ros_cloud.height, 
                            dtype=[('x', np.float32), 
                                   ('y', np.float32), 
                                   ('z', np.float32),
                                   ('rgb', np.float32)])
        
        # Read point cloud data
        is_bigendian = ros_cloud.is_bigendian
        point_step = ros_cloud.point_step
        
        # Get field offsets
        x_offset = next(field.offset for field in ros_cloud.fields if field.name == 'x')
        y_offset = next(field.offset for field in ros_cloud.fields if field.name == 'y')
        z_offset = next(field.offset for field in ros_cloud.fields if field.name == 'z')
        
        # Check if RGB/RGBA fields exist
        rgb_offset = None
        for field in ros_cloud.fields:
            if field.name in ['rgb', 'rgba']:
                rgb_offset = field.offset
                break
        
        # Extract points
        for i in range(ros_cloud.width * ros_cloud.height):
            start_idx = i * point_step
            
            # Extract XYZ
            x = struct.unpack('f', ros_cloud.data[start_idx + x_offset:start_idx + x_offset + 4])[0]
            y = struct.unpack('f', ros_cloud.data[start_idx + y_offset:start_idx + y_offset + 4])[0]
            z = struct.unpack('f', ros_cloud.data[start_idx + z_offset:start_idx + z_offset + 4])[0]
            
            cloud_data[i]['x'] = x
            cloud_data[i]['y'] = y
            cloud_data[i]['z'] = z
            
            # Extract RGB if available
            if rgb_offset is not None:
                rgb = struct.unpack('f', ros_cloud.data[start_idx + rgb_offset:start_idx + rgb_offset + 4])[0]
                cloud_data[i]['rgb'] = rgb
        
        # Create Open3D point cloud
        o3d_cloud = o3d.geometry.PointCloud()
        
        # Add points
        points = np.zeros((cloud_data.shape[0], 3))
        points[:, 0] = cloud_data['x']
        points[:, 1] = cloud_data['y']
        points[:, 2] = cloud_data['z']
        o3d_cloud.points = o3d.utility.Vector3dVector(points)
        
        # Add colors if available
        if rgb_offset is not None:
            colors = np.zeros((cloud_data.shape[0], 3))
            
            # Convert float RGB to separate R, G, B values (0-1)
            for i in range(cloud_data.shape[0]):
                rgb = int(cloud_data[i]['rgb'])
                r = ((rgb >> 16) & 0xFF) / 255.0
                g = ((rgb >> 8) & 0xFF) / 255.0
                b = (rgb & 0xFF) / 255.0
                colors[i] = [r, g, b]
            
            o3d_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        return o3d_cloud
    
    def o3d_to_ros(self, o3d_cloud, header):
        """Convert Open3D point cloud to ROS PointCloud2"""
        # Create PointCloud2 message
        ros_cloud = PointCloud2()
        
        # Set header
        ros_cloud.header = header
        
        # Get points and colors
        points = np.asarray(o3d_cloud.points)
        
        # Set dimensions
        ros_cloud.height = 1
        ros_cloud.width = points.shape[0]
        
        # Set fields
        ros_cloud.fields = [
            self.create_field('x', 0, 7, 1),
            self.create_field('y', 4, 7, 1),
            self.create_field('z', 8, 7, 1)
        ]
        
        # Check if colors are available
        has_colors = hasattr(o3d_cloud, 'colors') and o3d_cloud.colors
        
        if has_colors:
            colors = np.asarray(o3d_cloud.colors)
            ros_cloud.fields.append(self.create_field('rgb', 12, 7, 1))
            point_step = 16
        else:
            point_step = 12
        
        # Set point data
        data = np.zeros(points.shape[0] * point_step, dtype=np.uint8)
        
        for i in range(points.shape[0]):
            # Add XYZ
            data[i*point_step:i*point_step+4] = struct.pack('f', points[i, 0])
            data[i*point_step+4:i*point_step+8] = struct.pack('f', points[i, 1])
            data[i*point_step+8:i*point_step+12] = struct.pack('f', points[i, 2])
            
            # Add RGB if available
            if has_colors:
                r = int(colors[i, 0] * 255)
                g = int(colors[i, 1] * 255)
                b = int(colors[i, 2] * 255)
                rgb = (r << 16) | (g << 8) | b
                data[i*point_step+12:i*point_step+16] = struct.pack('f', rgb)
        
        ros_cloud.data = bytes(data)
        ros_cloud.point_step = point_step
        ros_cloud.row_step = ros_cloud.point_step * ros_cloud.width
        ros_cloud.is_dense = True
        
        return ros_cloud
    
    def create_field(self, name, offset, datatype, count):
        """Create a PointField for PointCloud2 message"""
        from sensor_msgs.msg import PointField
        field = PointField()
        field.name = name
        field.offset = offset
        field.datatype = datatype
        field.count = count
        return field
    
    def process_point_cloud(self, cloud, header):
        """Process point cloud to detect cylinders"""
        if len(cloud.points) == 0:
            self.get_logger().warn('Empty point cloud')
            return
        
        # Apply voxel grid downsampling
        cloud_filtered = cloud.voxel_down_sample(voxel_size=self.voxel_leaf_size)
        self.get_logger().debug(f'After voxel filtering: {len(cloud_filtered.points)} points')
        
        # Apply passthrough filter
        points = np.asarray(cloud_filtered.points)
        mask = np.ones(len(points), dtype=bool)
        
        # Filter by X range
        mask = np.logical_and(mask, points[:, 0] >= self.passthrough_x_min)
        mask = np.logical_and(mask, points[:, 0] <= self.passthrough_x_max)
        
        # Filter by Y range
        mask = np.logical_and(mask, points[:, 1] >= self.passthrough_y_min)
        mask = np.logical_and(mask, points[:, 1] <= self.passthrough_y_max)
        
        # Filter by Z range
        mask = np.logical_and(mask, points[:, 2] >= self.passthrough_z_min)
        mask = np.logical_and(mask, points[:, 2] <= self.passthrough_z_max)
        
        # Apply the filter
        cloud_filtered_passthrough = o3d.geometry.PointCloud()
        cloud_filtered_passthrough.points = o3d.utility.Vector3dVector(points[mask])
        
        # Copy colors if available
        if hasattr(cloud_filtered, 'colors') and cloud_filtered.colors:
            colors = np.asarray(cloud_filtered.colors)
            cloud_filtered_passthrough.colors = o3d.utility.Vector3dVector(colors[mask])
        
        self.get_logger().debug(f'After passthrough filtering: {len(cloud_filtered_passthrough.points)} points')
        
        # Apply statistical outlier removal
        if len(cloud_filtered_passthrough.points) > self.outlier_mean_k:
            cloud_filtered_outliers, _ = cloud_filtered_passthrough.remove_statistical_outlier(
                nb_neighbors=self.outlier_mean_k,
                std_ratio=self.outlier_std_dev
            )
            self.get_logger().debug(f'After outlier removal: {len(cloud_filtered_outliers.points)} points')
        else:
            cloud_filtered_outliers = cloud_filtered_passthrough
            self.get_logger().debug('Skipped outlier removal (not enough points)')
        
        # Estimate normals
        cloud_filtered_outliers.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius_search, 
                max_nn=30
            )
        )
        
        # Detect cylinders
        detected_cylinders = self.detect_cylinders(cloud_filtered_outliers)
        
        # Update tracking
        self.update_cylinder_tracking(detected_cylinders)
        
        self.get_logger().info(f'Detected {len(detected_cylinders)} cylinders')
        
        # Publish filtered point cloud
        filtered_cloud_msg = self.o3d_to_ros(cloud_filtered_outliers, header)
        self.filtered_cloud_pub.publish(filtered_cloud_msg)
        
        # Publish cylinder poses
        cylinder_poses = self.create_cylinder_poses(detected_cylinders, header)
        self.cylinder_poses_pub.publish(cylinder_poses)
        
        # Publish visualization markers
        markers = self.create_cylinder_markers(detected_cylinders, header)
        self.marker_pub.publish(markers)
        
        # Publish cylinder count
        count_msg = Int32()
        count_msg.data = len(detected_cylinders)
        self.cylinder_count_pub.publish(count_msg)
    
    def detect_cylinders(self, cloud):
        """Detect cylinders in point cloud using RANSAC"""
        detected_cylinders = []
        
        if len(cloud.points) < self.min_cylinder_points:
            self.get_logger().warn(f'Not enough points for cylinder detection: {len(cloud.points)}')
            return detected_cylinders
        
        # Get points and normals
        points = np.asarray(cloud.points)
        normals = np.asarray(cloud.normals)
        
        # Create a copy of the point cloud for segmentation
        remaining_points = points.copy()
        remaining_normals = normals.copy()
        
        # Keep finding cylinders until we run out of points or reach the maximum number
        max_cylinders = 10  # Limit to prevent infinite loops on noisy data
        
        for i in range(max_cylinders):
            if len(remaining_points) < self.min_cylinder_points:
                self.get_logger().debug(f'Not enough remaining points: {len(remaining_points)}')
                break
            
            # Use PyRANSAC3D for cylinder detection
            cylinder = pyrsc.Cylinder()
            
            try:
                # Perform RANSAC cylinder fitting
                center, axis, radius, inlier_indices = cylinder.fit(
                    remaining_points, 
                    thresh=self.ransac_threshold, 
                    maxIteration=self.ransac_iterations
                )
                
                # Check if we found a cylinder
                if len(inlier_indices) < self.min_cylinder_points:
                    self.get_logger().debug(f'Found cylinder with only {len(inlier_indices)} points, skipping')
                    break
                
                # Check radius limits
                if radius < self.cylinder_radius_min or radius > self.cylinder_radius_max:
                    self.get_logger().debug(f'Cylinder radius {radius:.3f} outside limits, skipping')
                    # Remove these points and continue
                    mask = np.ones(len(remaining_points), dtype=bool)
                    mask[inlier_indices] = False
                    remaining_points = remaining_points[mask]
                    remaining_normals = remaining_normals[mask]
                    continue
                
                # Compute confidence score
                confidence = self.compute_confidence(remaining_points, inlier_indices, center, axis, radius)
                
                # Check confidence threshold
                if confidence < self.min_confidence_threshold:
                    self.get_logger().debug(f'Cylinder confidence {confidence:.2f} below threshold, skipping')
                    break
                
                # Create cylinder detection
                cylinder_detection = CylinderDetection(
                    center=np.array(center),
                    axis=np.array(axis),
                    radius=radius,
                    confidence=confidence,
                    point_count=len(inlier_indices),
                    tracking_id=-1,  # Will be assigned in tracking
                    age=0            # Will be set in tracking
                )
                
                detected_cylinders.append(cylinder_detection)
                self.get_logger().info(
                    f'Found cylinder at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) '
                    f'with radius {radius:.3f} m, confidence {confidence:.2f}, {len(inlier_indices)} points'
                )
                
                # Remove inlier points from remaining cloud
                mask = np.ones(len(remaining_points), dtype=bool)
                mask[inlier_indices] = False
                remaining_points = remaining_points[mask]
                remaining_normals = remaining_normals[mask]
                
                self.get_logger().debug(f'Remaining points: {len(remaining_points)}')
                
            except Exception as e:
                self.get_logger().error(f'Error in cylinder detection: {str(e)}')
                break
        
        return detected_cylinders
    
    def compute_confidence(self, points, inlier_indices, center, axis, radius):
        """Compute confidence score for cylinder detection"""
        # Basic confidence based on number of points
        point_ratio = len(inlier_indices) / len(points)
        
        # Compute mean distance from points to the cylinder model
        total_distance = 0.0
        center_np = np.array(center)
        axis_np = np.array(axis)
        axis_np = axis_np / np.linalg.norm(axis_np)  # Normalize
        
        for idx in inlier_indices:
            point = points[idx]
            
            # Vector from point to center
            point_to_center = point - center_np
            
            # Project point onto axis
            t = np.dot(point_to_center, axis_np)
            projection = center_np + t * axis_np
            
            # Compute distance from point to projection (should be close to radius)
            distance = np.linalg.norm(point - projection)
            distance_error = abs(distance - radius)
            
            total_distance += distance_error
        
        mean_distance_error = total_distance / len(inlier_indices)
        distance_score = 1.0 - min(1.0, mean_distance_error / radius)
        
        # Combine scores
        confidence = 0.7 * point_ratio + 0.3 * distance_score
        
        return confidence
    
    def update_cylinder_tracking(self, current_cylinders):
        """Track cylinders across frames"""
        # If no previous cylinders, just assign new IDs to all current ones
        if len(self.previous_cylinders) == 0:
            for cylinder in current_cylinders:
                cylinder.tracking_id = self.next_tracking_id
                self.next_tracking_id += 1
                cylinder.age = 1
            
            self.previous_cylinders = current_cylinders
            return
        
        # Create a copy of previous cylinders for matching
        unmatched_previous = self.previous_cylinders.copy()
        
        # For each current cylinder, try to find a match in previous ones
        for current in current_cylinders:
            found_match = False
            
            for i, previous in enumerate(unmatched_previous):
                # Compute distance between cylinder centers
                distance = np.linalg.norm(current.center - previous.center)
                
                # If close enough, consider it the same cylinder
                if distance < self.tracking_distance_threshold:
                    current.tracking_id = previous.tracking_id
                    current.age = previous.age + 1
                    found_match = True
                    
                    # Remove this previous cylinder from unmatched list
                    unmatched_previous.pop(i)
                    break
            
            # If no match found, assign a new ID
            if not found_match:
                current.tracking_id = self.next_tracking_id
                self.next_tracking_id += 1
                current.age = 1
        
        # Update previous cylinders for next frame
        self.previous_cylinders = current_cylinders
    
    def create_cylinder_poses(self, cylinders, header):
        """Create PoseArray message for detected cylinders"""
        pose_array = PoseArray()
        pose_array.header = header
        
        for cylinder in cylinders:
            pose = Pose()
            
            # Set position
            pose.position.x = float(cylinder.center[0])
            pose.position.y = float(cylinder.center[1])
            pose.position.z = float(cylinder.center[2])
            
            # Set orientation (convert axis direction to quaternion)
            # Default cylinder axis is along Z, so we need to rotate to match detected axis
            z_axis = np.array([0, 0, 1])
            norm_axis = cylinder.axis / np.linalg.norm(cylinder.axis)
            
            # Check if axes are not parallel
            if not np.allclose(np.abs(np.dot(z_axis, norm_axis)), 1.0):
                # Compute rotation from z-axis to cylinder axis
                rotation_axis = np.cross(z_axis, norm_axis)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                
                angle = np.arccos(np.dot(z_axis, norm_axis))
                
                # Create rotation matrix
                rotation = R.from_rotvec(rotation_axis * angle)
                quat = rotation.as_quat()  # [x, y, z, w]
                
                pose.orientation.x = float(quat[0])
                pose.orientation.y = float(quat[1])
                pose.orientation.z = float(quat[2])
                pose.orientation.w = float(quat[3])
            else:
                # If axes are parallel, use identity quaternion or flip if pointing in opposite direction
                if np.dot(z_axis, norm_axis) > 0:
                    pose.orientation.w = 1.0  # Identity quaternion
                else:
                    # 180 degree rotation around X axis
                    pose.orientation.x = 1.0
                    pose.orientation.w = 0.0
            
            pose_array.poses.append(pose)
        
        return pose_array
    
    def create_cylinder_markers(self, cylinders, header):
        """Create visualization markers for detected cylinders"""
        marker_array = MarkerArray()
        
        for i, cylinder in enumerate(cylinders):
            # Create cylinder marker
            marker = Marker()
            marker.header = header
            marker.ns = "cylinders"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position.x = float(cylinder.center[0])
            marker.pose.position.y = float(cylinder.center[1])
            marker.pose.position.z = float(cylinder.center[2])
            
            # Set orientation (same as in pose array)
            z_axis = np.array([0, 0, 1])
            norm_axis = cylinder.axis / np.linalg.norm(cylinder.axis)
            
            # Check if axes are not parallel
            if not np.allclose(np.abs(np.dot(z_axis, norm_axis)), 1.0):
                rotation_axis = np.cross(z_axis, norm_axis)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                
                angle = np.arccos(np.dot(z_axis, norm_axis))
                
                rotation = R.from_rotvec(rotation_axis * angle)
                quat = rotation.as_quat()
                
                marker.pose.orientation.x = float(quat[0])
                marker.pose.orientation.y = float(quat[1])
                marker.pose.orientation.z = float(quat[2])
                marker.pose.orientation.w = float(quat[3])
            else:
                if np.dot(z_axis, norm_axis) > 0:
                    marker.pose.orientation.w = 1.0
                else:
                    marker.pose.orientation.x = 1.0
                    marker.pose.orientation.w = 0.0
            
            # Set scale
            marker.scale.x = cylinder.radius * 2  # Diameter in x
            marker.scale.y = cylinder.radius * 2  # Diameter in y
            marker.scale.z = 0.2  # Height (arbitrary, for visualization)
            
            # Set color based on confidence
            marker.color.r = 1.0 - cylinder.confidence
            marker.color.g = cylinder.confidence
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 500000000  # 0.5 seconds
            
            marker_array.markers.append(marker)
            
            # Add text marker with cylinder info
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "cylinder_info"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(cylinder.center[0])
            text_marker.pose.position.y = float(cylinder.center[1])
            text_marker.pose.position.z = float(cylinder.center[2]) + 0.15  # Position text above cylinder
            text_marker.pose.orientation.w = 1.0
            
            text_marker.text = (
                f"Cyl #{cylinder.tracking_id}\n"
                f"R={cylinder.radius*100:.1f}cm\n"
                f"C={cylinder.confidence:.2f}\n"
                f"Age={cylinder.age}"
            )
            
            text_marker.scale.z = 0.05  # Text size
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 0.8
            
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = 500000000  # 0.5 seconds
            
            marker_array.markers.append(text_marker)
        
        return marker_array
    
    def depth_image_callback(self, msg):
        """Process incoming depth image messages"""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CV bridge exception: {str(e)}')
    
    def rgb_image_callback(self, msg):
        """Process incoming RGB image messages"""
        try:
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CV bridge exception: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    detector = CylinderDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
