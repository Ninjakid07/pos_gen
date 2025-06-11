#!/usr/bin/env python3
"""
Tree Row Navigator for pos_gen package
ROSbot 3 Pro Tree Row Navigation System
Implements LiDAR-based wall-following for macadamia field navigation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import numpy as np
import math
from enum import Enum
import time

class NavigationState(Enum):
    INIT = "INITIALIZING"
    FIND_ROW = "FINDING_ROW"
    FOLLOW_ROW = "FOLLOWING_ROW"
    END_DETECTED = "END_DETECTED"
    STOPPED = "STOPPED"
    EMERGENCY = "EMERGENCY_STOP"

class TreeRowNavigator(Node):
    def __init__(self):
        super().__init__('tree_row_navigator')
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.state_pub = self.create_publisher(String, '/navigation_state', 10)
        
        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Navigation parameters
        self.DESIRED_DISTANCE = 0.5  # meters from tree line
        self.MAX_SPEED = 0.2  # m/s - conservative speed
        self.MIN_TREE_DISTANCE = 0.3  # minimum safe distance
        self.MAX_TREE_DISTANCE = 1.5  # maximum detection range
        self.TREE_DETECTION_THRESHOLD = 0.1  # clustering threshold
        self.END_DETECTION_TIME = 2.0  # seconds to confirm row end
        self.FIELD_BOUNDARY = 1.8  # field size minus safety margin
        
        # Control parameters
        self.KP_DISTANCE = 2.0  # proportional gain for distance control
        self.KP_ANGLE = 1.5     # proportional gain for angle control
        self.MAX_ANGULAR_VEL = 0.5  # rad/s
        
        # State variables
        self.state = NavigationState.INIT
        self.last_scan = None
        self.trees_on_right = True  # assume trees on right side initially
        self.state_start_time = time.time()
        self.end_detection_start = None
        self.robot_x = 0.0  # simple odometry tracking
        self.robot_y = 0.0
        self.robot_theta = 0.0
        
        # Timer for control loop
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz
        
        self.get_logger().info("Tree Row Navigator initialized")
        self.publish_state()

    def scan_callback(self, msg):
        """Process LiDAR scan data"""
        self.last_scan = msg
        
    def get_tree_line_info(self):
        """
        Analyze LiDAR data to find tree line
        Returns: (tree_detected, distance_to_trees, angle_to_trees, front_clear)
        """
        if self.last_scan is None:
            return False, 0.0, 0.0, True
            
        ranges = np.array(self.last_scan.ranges)
        angles = np.linspace(self.last_scan.angle_min, 
                           self.last_scan.angle_max, 
                           len(ranges))
        
        # Remove invalid readings
        valid_mask = np.isfinite(ranges) & (ranges > 0.1)
        ranges = ranges[valid_mask]
        angles = angles[valid_mask]
        
        if len(ranges) == 0:
            return False, 0.0, 0.0, True
        
        # Check front clearance (±30 degrees ahead)
        front_mask = np.abs(angles) < math.pi/6
        front_ranges = ranges[front_mask]
        front_clear = len(front_ranges) == 0 or np.min(front_ranges) > 0.8
        
        # Look for trees on the right side (270° ± 45°)
        if self.trees_on_right:
            side_mask = (angles > math.pi - math.pi/4) & (angles < math.pi + math.pi/4)
        else:
            # Look on left side (90° ± 45°)
            side_mask = (angles > math.pi/2 - math.pi/4) & (angles < math.pi/2 + math.pi/4)
        
        side_ranges = ranges[side_mask]
        side_angles = angles[side_mask]
        
        # Filter for tree detection range
        tree_mask = (side_ranges > self.MIN_TREE_DISTANCE) & \
                   (side_ranges < self.MAX_TREE_DISTANCE)
        tree_ranges = side_ranges[tree_mask]
        tree_angles = side_angles[tree_mask]
        
        if len(tree_ranges) == 0:
            return False, 0.0, 0.0, front_clear
        
        # Cluster nearby points (simple clustering)
        tree_clusters = self.cluster_points(tree_ranges, tree_angles)
        
        if len(tree_clusters) == 0:
            return False, 0.0, 0.0, front_clear
        
        # Find closest tree cluster
        closest_cluster = min(tree_clusters, key=lambda c: c['distance'])
        
        return True, closest_cluster['distance'], closest_cluster['angle'], front_clear
    
    def cluster_points(self, ranges, angles):
        """Simple clustering of LiDAR points to identify trees"""
        if len(ranges) == 0:
            return []
        
        # Convert to cartesian coordinates
        points = []
        for r, a in zip(ranges, angles):
            x = r * math.cos(a)
            y = r * math.sin(a)
            points.append({'x': x, 'y': y, 'range': r, 'angle': a})
        
        clusters = []
        used = [False] * len(points)
        
        for i, point in enumerate(points):
            if used[i]:
                continue
                
            cluster_points = [point]
            used[i] = True
            
            # Find nearby points
            for j, other_point in enumerate(points):
                if used[j]:
                    continue
                    
                dist = math.sqrt((point['x'] - other_point['x'])**2 + 
                               (point['y'] - other_point['y'])**2)
                
                if dist < self.TREE_DETECTION_THRESHOLD:
                    cluster_points.append(other_point)
                    used[j] = True
            
            # Only consider clusters with enough points (trees are substantial)
            if len(cluster_points) >= 3:
                # Calculate cluster center
                avg_x = sum(p['x'] for p in cluster_points) / len(cluster_points)
                avg_y = sum(p['y'] for p in cluster_points) / len(cluster_points)
                avg_range = math.sqrt(avg_x**2 + avg_y**2)
                avg_angle = math.atan2(avg_y, avg_x)
                
                clusters.append({
                    'distance': avg_range,
                    'angle': avg_angle,
                    'points': len(cluster_points)
                })
        
        return clusters
    
    def control_loop(self):
        """Main control loop - state machine execution"""
        current_time = time.time()
        cmd = Twist()
        
        # Get sensor information
        tree_detected, distance, angle, front_clear = self.get_tree_line_info()
        
        # Emergency stop conditions
        if not front_clear or self.check_boundary_collision():
            self.state = NavigationState.EMERGENCY
        
        # State machine
        if self.state == NavigationState.INIT:
            self.handle_init_state(cmd, current_time)
            
        elif self.state == NavigationState.FIND_ROW:
            self.handle_find_row_state(cmd, tree_detected, distance, angle)
            
        elif self.state == NavigationState.FOLLOW_ROW:
            self.handle_follow_row_state(cmd, tree_detected, distance, angle, front_clear)
            
        elif self.state == NavigationState.END_DETECTED:
            self.handle_end_detected_state(cmd, current_time)
            
        elif self.state == NavigationState.STOPPED:
            self.handle_stopped_state(cmd)
            
        elif self.state == NavigationState.EMERGENCY:
            self.handle_emergency_state(cmd)
        
        # Publish commands and state
        self.cmd_pub.publish(cmd)
        self.publish_state()
        
        # Simple dead reckoning for boundary checking
        self.update_odometry(cmd.linear.x, cmd.angular.z)
    
    def handle_init_state(self, cmd, current_time):
        """Initialize robot - rotate slowly to find trees"""
        cmd.angular.z = 0.3  # slow rotation
        
        tree_detected, distance, angle, _ = self.get_tree_line_info()
        
        if tree_detected:
            self.get_logger().info(f"Trees detected at distance {distance:.2f}m")
            self.state = NavigationState.FIND_ROW
            self.state_start_time = current_time
        elif current_time - self.state_start_time > 10.0:  # timeout
            self.get_logger().warn("Tree detection timeout - stopping")
            self.state = NavigationState.EMERGENCY
    
    def handle_find_row_state(self, cmd, tree_detected, distance, angle):
        """Position robot parallel to tree row"""
        if not tree_detected:
            self.state = NavigationState.INIT
            return
        
        # Align with tree line
        if abs(angle) > 0.1:  # not parallel enough
            cmd.angular.z = -self.KP_ANGLE * angle
        else:
            # Move to desired distance
            distance_error = distance - self.DESIRED_DISTANCE
            if abs(distance_error) > 0.1:
                if distance_error > 0:  # too far, move closer
                    cmd.linear.x = 0.1
                    if self.trees_on_right:
                        cmd.angular.z = -0.2
                    else:
                        cmd.angular.z = 0.2
                else:  # too close, move away
                    cmd.linear.x = 0.1
                    if self.trees_on_right:
                        cmd.angular.z = 0.2
                    else:
                        cmd.angular.z = -0.2
            else:
                # Good position and alignment - start following
                self.get_logger().info("Starting row following")
                self.state = NavigationState.FOLLOW_ROW
    
    def handle_follow_row_state(self, cmd, tree_detected, distance, angle, front_clear):
        """Follow the tree row maintaining desired distance"""
        if not tree_detected:
            # Start end detection timer
            if self.end_detection_start is None:
                self.end_detection_start = time.time()
                self.get_logger().info("No trees detected - starting end detection timer")
            elif time.time() - self.end_detection_start > self.END_DETECTION_TIME:
                self.get_logger().info("Row end confirmed")
                self.state = NavigationState.END_DETECTED
                return
        else:
            # Reset end detection timer
            self.end_detection_start = None
        
        if not front_clear:
            self.get_logger().warn("Front blocked - stopping")
            self.state = NavigationState.END_DETECTED
            return
        
        # Wall following control
        if tree_detected:
            # Distance control
            distance_error = distance - self.DESIRED_DISTANCE
            angular_correction = self.KP_DISTANCE * distance_error
            
            # Angle control (keep parallel to tree line)
            angle_correction = self.KP_ANGLE * angle
            
            # Combine corrections
            if self.trees_on_right:
                cmd.angular.z = angular_correction - angle_correction
            else:
                cmd.angular.z = -angular_correction - angle_correction
            
            # Limit angular velocity
            cmd.angular.z = max(-self.MAX_ANGULAR_VEL, 
                              min(self.MAX_ANGULAR_VEL, cmd.angular.z))
            
            # Forward movement
            cmd.linear.x = self.MAX_SPEED
        
        # Safety check - if too close to trees, slow down and turn away
        if tree_detected and distance < self.MIN_TREE_DISTANCE:
            cmd.linear.x = 0.05
            if self.trees_on_right:
                cmd.angular.z = 0.3  # turn left
            else:
                cmd.angular.z = -0.3  # turn right
    
    def handle_end_detected_state(self, cmd, current_time):
        """Gracefully stop at row end"""
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.state = NavigationState.STOPPED
        self.get_logger().info("Navigation completed - robot stopped")
    
    def handle_stopped_state(self, cmd):
        """Robot stopped - all commands zero"""
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
    
    def handle_emergency_state(self, cmd):
        """Emergency stop - all movement ceased"""
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.get_logger().error("EMERGENCY STOP ACTIVATED")
    
    def check_boundary_collision(self):
        """Check if robot is approaching field boundaries"""
        boundary_margin = 0.2  # safety margin
        
        if (abs(self.robot_x) > self.FIELD_BOUNDARY - boundary_margin or 
            abs(self.robot_y) > self.FIELD_BOUNDARY - boundary_margin):
            self.get_logger().warn(f"Approaching boundary at ({self.robot_x:.2f}, {self.robot_y:.2f})")
            return True
        return False
    
    def update_odometry(self, linear_vel, angular_vel):
        """Simple dead reckoning for boundary checking"""
        dt = 0.1  # control loop period
        
        self.robot_theta += angular_vel * dt
        self.robot_x += linear_vel * math.cos(self.robot_theta) * dt
        self.robot_y += linear_vel * math.sin(self.robot_theta) * dt
    
    def publish_state(self):
        """Publish current navigation state"""
        state_msg = String()
        state_msg.data = self.state.value
        self.state_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    
    navigator = TreeRowNavigator()
    
    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        navigator.get_logger().info("Navigation interrupted by user")
    finally:
        # Send stop command
        stop_cmd = Twist()
        navigator.cmd_pub.publish(stop_cmd)
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
