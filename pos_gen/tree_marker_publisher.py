#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# ============== EASILY CONFIGURABLE PARAMETERS ==============
NUM_ROWS = 3          # Number of tree rows
TREES_PER_ROW = 5     # Number of trees in each row
TREE_SPACING = 0.3    # Distance between trees in a row (meters)
ROW_SPACING = 0.5     # Distance between rows (meters)

# Tree marker appearance
TREE_RADIUS = 0.05    # Radius of tree trunk (meters)
TREE_HEIGHT = 0.5     # Height of tree marker (meters)
# ==========================================================

class TreeMarkerPublisher(Node):
    def __init__(self):
        super().__init__('tree_marker_publisher')
        
        # Publisher for tree positions
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_pos', 10)
        
        # TF2 buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Wait a moment for transforms to become available
        self.create_timer(2.0, self.initialize_markers)  # One-shot timer for initialization
        
        # Publishing timer
        self.publish_timer = None
        self.marker_array = MarkerArray()
        
        self.get_logger().info(f'Tree marker publisher initialized with {NUM_ROWS} rows and {TREES_PER_ROW} trees per row')
        
    def initialize_markers(self):
        """Initialize markers after waiting for transforms"""
        # Cancel the initialization timer
        self.destroy_timer(self._timers[0])
        
        # Generate markers
        self.marker_array = self.generate_tree_markers()
        
        # Start publishing
        self.publish_timer = self.create_timer(1.0, self.publish_markers)  # 1 Hz
        
    def generate_tree_markers(self):
        """Generate tree markers in a grid pattern relative to robot's initial position"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # Get robot's current position in odom frame
        try:
            # Try to get transform from base_link to odom
            transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            self.get_logger().info(f'Robot initial position: x={robot_x:.2f}, y={robot_y:.2f}')
        except TransformException as ex:
            self.get_logger().warn(f'Could not get robot transform: {ex}, using (0,0) as origin')
            robot_x = 0.0
            robot_y = 0.0
        
        # Generate markers for each tree
        for row in range(NUM_ROWS):
            for tree in range(TREES_PER_ROW):
                marker = Marker()
                
                # Basic marker properties
                marker.header.frame_id = "odom"  # Fixed frame so markers don't move
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "trees"
                marker.id = marker_id
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                # Position relative to robot's initial position
                # Trees are arranged with rows going forward (x) and trees in a row going sideways (y)
                marker.pose.position.x = robot_x + (tree * TREE_SPACING)
                marker.pose.position.y = robot_y + (row * ROW_SPACING)
                marker.pose.position.z = TREE_HEIGHT / 2  # Cylinder centered at half height
                
                # Orientation (standing upright)
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # Size
                marker.scale.x = TREE_RADIUS * 2  # Diameter
                marker.scale.y = TREE_RADIUS * 2  # Diameter
                marker.scale.z = TREE_HEIGHT
                
                # Color (brown for tree trunk)
                marker.color.r = 0.55
                marker.color.g = 0.27
                marker.color.b = 0.07
                marker.color.a = 1.0
                
                # Lifetime (0 means forever)
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 0
                
                marker_array.markers.append(marker)
                marker_id += 1
                
        self.get_logger().info(f'Generated {marker_id} tree markers')
        return marker_array
    
    def publish_markers(self):
        """Publish the marker array"""
        # Update timestamp for all markers
        current_time = self.get_clock().now().to_msg()
        for marker in self.marker_array.markers:
            marker.header.stamp = current_time
        
        self.marker_pub.publish(self.marker_array)

def main(args=None):
    rclpy.init(args=args)
    
    tree_marker_publisher = TreeMarkerPublisher()
    
    try:
        rclpy.spin(tree_marker_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        tree_marker_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
