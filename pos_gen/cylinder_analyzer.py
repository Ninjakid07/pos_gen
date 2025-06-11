#!/usr/bin/env python3
"""
Simple Cylinder Detector for ROSbot 3 Pro
Detects the largest cylinder in the environment and reports its properties
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class SimpleCylinderDetector(Node):
    def __init__(self):
        super().__init__('simple_cylinder_detector')
        
        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Detection parameters (optimized from your test data)
        self.MIN_RANGE = 0.1
        self.MAX_RANGE = 2.0
        self.CLUSTERING_THRESHOLD = 0.08  # 8cm clustering
        self.MIN_CLUSTER_POINTS = 3
        self.MAX_CLUSTER_POINTS = 50  # Filter out walls
        self.MIN_CONFIDENCE_THRESHOLD = 0.7
        
        self.get_logger().info("Simple Cylinder Detector started")
        self.get_logger().info("Looking for the largest cylinder in the environment...")

    def scan_callback(self, msg):
        """Process LiDAR scan and detect cylinders"""
        # Convert scan to numpy arrays
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Remove invalid readings
        valid_mask = np.isfinite(ranges) & (ranges >= self.MIN_RANGE) & (ranges <= self.MAX_RANGE)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return
        
        # Convert to cartesian coordinates
        points = []
        for r, a in zip(valid_ranges, valid_angles):
            x = r * math.cos(a)
            y = r * math.sin(a)
            points.append({'x': x, 'y': y, 'range': r, 'angle': a})
        
        # Cluster points
        clusters = self.cluster_points(points)
        
        # Analyze clusters as potential cylinders
        cylinder_candidates = []
        for cluster in clusters:
            cylinder_analysis = self.analyze_as_cylinder(cluster)
            if cylinder_analysis['is_cylinder']:
                cylinder_candidates.append(cylinder_analysis)
        
        # Find the largest cylinder (highest radius)
        if cylinder_candidates:
            largest_cylinder = max(cylinder_candidates, key=lambda c: c['radius'])
            
            # Only report if confidence is high enough
            if largest_cylinder['confidence'] >= self.MIN_CONFIDENCE_THRESHOLD:
                self.get_logger().info(
                    f"Cylinder detected at {largest_cylinder['distance']:.2f}m, "
                    f"confidence: {largest_cylinder['confidence']:.2f}, "
                    f"radius: {largest_cylinder['radius']*100:.1f}cm"
                )

    def cluster_points(self, points):
        """Cluster LiDAR points using distance-based grouping"""
        if len(points) == 0:
            return []
        
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
                
                if dist < self.CLUSTERING_THRESHOLD:
                    cluster_points.append(other_point)
                    used[j] = True
            
            # Filter by point count (exclude walls, include cylinders)
            if self.MIN_CLUSTER_POINTS <= len(cluster_points) <= self.MAX_CLUSTER_POINTS:
                cluster = self.calculate_cluster_properties(cluster_points)
                clusters.append(cluster)
        
        return clusters

    def calculate_cluster_properties(self, cluster_points):
        """Calculate cluster properties"""
        # Center calculation
        center_x = sum(p['x'] for p in cluster_points) / len(cluster_points)
        center_y = sum(p['y'] for p in cluster_points) / len(cluster_points)
        center_range = math.sqrt(center_x**2 + center_y**2)
        center_angle = math.atan2(center_y, center_x)
        
        # Calculate cluster span (diameter)
        max_span = 0.0
        for i, p1 in enumerate(cluster_points):
            for j, p2 in enumerate(cluster_points[i+1:], i+1):
                dist = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                max_span = max(max_span, dist)
        
        # Range statistics for compactness
        ranges = [p['range'] for p in cluster_points]
        range_std = np.std(ranges)
        
        return {
            'points': len(cluster_points),
            'center_x': center_x,
            'center_y': center_y,
            'center_range': center_range,
            'center_angle': center_angle,
            'span': max_span,
            'range_std': range_std
        }

    def analyze_as_cylinder(self, cluster):
        """Analyze cluster as potential cylinder"""
        # Estimate radius (half of span)
        radius = cluster['span'] / 2.0
        
        # Size validation (reasonable cylinder size 1-10cm radius)
        size_score = 0.0
        if 0.01 <= radius <= 0.10:  # 1-10cm radius
            # Higher score for larger radius (since you said yours is the largest)
            size_score = min(1.0, radius / 0.05)  # Max score at 5cm radius
        
        # Point count validation (based on your test data: 17-44 points typical)
        point_score = 0.0
        if 5 <= cluster['points'] <= 50:
            if 10 <= cluster['points'] <= 40:  # Optimal range
                point_score = 1.0
            else:
                point_score = 0.7
        
        # Compactness (cylinders should be compact)
        compactness_score = 0.0
        if cluster['range_std'] < 0.1:  # Low range variation = compact
            compactness_score = 1.0
        elif cluster['range_std'] < 0.2:
            compactness_score = 0.7
        else:
            compactness_score = 0.3
        
        # Distance preference (closer objects easier to analyze)
        distance_score = 0.0
        if 0.3 <= cluster['center_range'] <= 1.5:
            if 0.5 <= cluster['center_range'] <= 1.0:
                distance_score = 1.0  # Optimal range
            else:
                distance_score = 0.8
        else:
            distance_score = 0.5
        
        # Combined confidence score (weighted for largest cylinder detection)
        confidence = (size_score * 0.4 +      # Emphasize size (largest cylinder)
                     point_score * 0.3 + 
                     compactness_score * 0.2 + 
                     distance_score * 0.1)
        
        # Determine if this is a cylinder
        is_cylinder = (confidence >= self.MIN_CONFIDENCE_THRESHOLD and 
                      radius >= 0.015)  # At least 1.5cm radius
        
        return {
            'is_cylinder': is_cylinder,
            'confidence': confidence,
            'distance': cluster['center_range'],
            'radius': radius,
            'diameter': radius * 2.0,
            'points': cluster['points'],
            'compactness': cluster['range_std']
        }

def main(args=None):
    rclpy.init(args=args)
    
    detector = SimpleCylinderDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info("Cylinder detection stopped")
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
