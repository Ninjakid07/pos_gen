#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import tf2_ros
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import math

# ============== EASILY CONFIGURABLE PARAMETERS ==============
TREES_PER_ROW = 10     # Number of trees in the row
TREE_SPACING = 0.2     # Distance between trees in a row (20 cm)
REVEAL_DISTANCE = 0.3  # Distance to tree before revealing next one (meters)

# Tree marker appearance
TREE_RADIUS = 0.05    # Radius of tree trunk (meters)
TREE_HEIGHT = 0.5     # Height of tree marker (meters)
# ==========================================================

class TreeMarkerPublisher(Node):
    def __init__(self):
        super().__init__('tree_marker_publisher')
        
        # Publisher for tree positions
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_pos', 10)
        
        # Subscribe to robot odometry for position tracking
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # TF2 buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # State variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.trees_revealed = 1  # Start with first tree visible
        self.initial_robot_x = None
        self.initial_robot_y = None
        self.tree_positions = []  # Store all tree positions
        self.marker_array = MarkerArray()
        
        # Wait for initial position
        self.create_timer(2.0, self.initialize_markers)  # One-shot timer for initialization
        
        # Publishing timer
        self.publish_timer = None
        
        self.get_logger().info(f'Tree marker publisher initialized with {TREES_PER_ROW} trees in a single row')
        
    def initialize_markers(self):
        """Initialize tree positions after waiting for transforms"""
        # Cancel the initialization timer
        self.destroy_timer(self._timers[0])
        
        # Get robot's initial position
        try:
            transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            self.initial_robot_x = transform.transform.translation.x
            self.initial_robot_y = transform.transform.translation.y
            self.get_logger().info(f'Robot initial position: x={self.initial_robot_x:.2f}, y={self.initial_robot_y:.2f}')
        except TransformException as ex:
            self.get_logger().warn(f'Could not get robot transform: {ex}, using (0,0) as origin')
            self.initial_robot_x = 0.0
            self.initial_robot_y = 0.0
        
        # Generate all tree positions (but don't create markers yet)
        self.generate_tree_positions()
        
        # Create first tree marker
        self.update_visible_trees()
        
        # Start publishing
        self.publish_timer = self.create_timer(0.1, self.publish_markers)  # 10 Hz for responsive updates
        
    def generate_tree_positions(self):
        """Generate positions for all trees in the row"""
        for tree_idx in range(TREES_PER_ROW):
            # Trees arranged in a straight line along X axis
            x = self.initial_robot_x + (tree_idx * TREE_SPACING)
            y = self.initial_robot_y
            self.tree_positions.append((x, y))
        
        self.get_logger().info(f'Generated positions for {TREES_PER_ROW} trees')
    
    def odom_callback(self, msg):
        """Update robot position from odometry"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Check if we should reveal more trees
        if self.trees_revealed < TREES_PER_ROW:
            # Get current visible tree position (last revealed tree)
            current_tree_x, current_tree_y = self.tree_positions[self.trees_revealed - 1]
            
            # Calculate distance to current tree
            distance = math.sqrt((self.robot_x - current_tree_x)**2 + (self.robot_y - current_tree_y)**2)
            
            # If close enough, reveal next tree
            if distance < REVEAL_DISTANCE:
                self.trees_revealed += 1
                self.update_visible_trees()
                self.get_logger().info(f'Revealed tree {self.trees_revealed}/{TREES_PER_ROW}')
    
    def create_tree_marker(self, tree_idx, x, y):
        """Create a single tree marker"""
        marker = Marker()
        
        # Basic marker properties
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trees"
        marker.id = tree_idx
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = TREE_HEIGHT / 2
        
        # Orientation (standing upright)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Size
        marker.scale.x = TREE_RADIUS * 2
        marker.scale.y = TREE_RADIUS * 2
        marker.scale.z = TREE_HEIGHT
        
        # Color (brown for tree trunk)
        marker.color.r = 0.55
        marker.color.g = 0.27
        marker.color.b = 0.07
        marker.color.a = 1.0
        
        # Lifetime (0 means forever)
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        
        return marker
    
    def update_visible_trees(self):
        """Update the marker array with currently visible trees"""
        self.marker_array.markers.clear()
        
        # Add markers for all revealed trees
        for i in range(self.trees_revealed):
            x, y = self.tree_positions[i]
            marker = self.create_tree_marker(i, x, y)
            self.marker_array.markers.append(marker)
    
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
