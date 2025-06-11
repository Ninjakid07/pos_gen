#!/usr/bin/env python3
"""
Multi-Cylinder Detector for ROSbot 3 Pro
Detects all cylinders within 70cm directly in front of the robot
Reports count, details, and identifies the largest cylinder
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import math

class MultiCylinderDetector(Node):
    def __init__(self):
        super().__init__('multi_cylinder_detector')
        
        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Detection parameters - optimized for close range
        self.MIN_RANGE = 0.1        # 10cm minimum
        self.MAX_RANGE = 0.7        # 70cm maximum
        self.FRONT_ANGLE_RANGE = math.pi/4  # ±45 degrees front sector
        
        # Clustering parameters
        self.CLUSTERING_THRESHOLD = 0.06  # 6cm clustering distance
        self.MIN_CLUSTER_POINTS = 8       # Minimum 8 points for cylinder
        self.MAX_CLUSTER_POINTS = 25      # Maximum 25 points for cylinder
        
        # Cylinder validation parameters
        self.MIN_DIAMETER = 0.04          # 4cm minimum diameter
        self.MAX_DIAMETER = 0.06          # 6cm maximum diameter
        self.MIN_ANGULAR_SPAN = 4.0       # 4 degrees minimum
        self.MAX_ANGULAR_SPAN = 8.0       # 8 degrees maximum
        self.MIN_CONFIDENCE = 0.9         # High confidence threshold
        self.GEOMETRY_TOLERANCE = 2.0     # ±2 degrees tolerance
        
        self.get_logger().info("Multi-Cylinder Detector started")
        self.get_logger().info("Detection range: 10-70cm, front sector: ±45°")
        self.get_logger().info("Looking for cylinders with 4-6cm diameter...")

    def scan_callback(self, msg):
        """Process LiDAR scan and detect all cylinders in front sector"""
        # Get front sector data only
        front_ranges, front_angles = self.extract_front_sector(msg)
        
        if len(front_ranges) == 0:
            return
        
        # Convert to cartesian coordinates
        points = []
        for r, a in zip(front_ranges, front_angles):
            x = r * math.cos(a)
            y = r * math.sin(a)
            points.append({'x': x, 'y': y, 'range': r, 'angle': a})
        
        # Cluster points to find distinct objects
        clusters = self.cluster_points(points)
        
        # Validate each cluster as potential cylinder
        valid_cylinders = []
        for cluster in clusters:
            cylinder_analysis = self.analyze_as_cylinder(cluster)
            if cylinder_analysis['is_cylinder']:
                valid_cylinders.append(cylinder_analysis)
        
        # Report results
        self.report_cylinder_detections(valid_cylinders)

    def extract_front_sector(self, msg):
        """Extract LiDAR data from front sector (±45°) within 70cm range"""
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Front sector mask: ±45 degrees from forward direction (0°)
        front_mask = np.abs(angles) <= self.FRONT_ANGLE_RANGE
        
        # Range mask: 10cm to 70cm
        range_mask = (ranges >= self.MIN_RANGE) & (ranges <= self.MAX_RANGE)
        
        # Valid data mask: finite values
        valid_mask = np.isfinite(ranges)
        
        # Combine all masks
        sector_mask = front_mask & range_mask & valid_mask
        
        return ranges[sector_mask], angles[sector_mask]

    def cluster_points(self, points):
        """Cluster points to identify distinct objects"""
        if len(points) == 0:
            return []
        
        clusters = []
        used = [False] * len(points)
        
        for i, point in enumerate(points):
            if used[i]:
                continue
                
            cluster_points = [point]
            used[i] = True
            
            # Find all nearby points for this cluster
            for j, other_point in enumerate(points):
                if used[j]:
                    continue
                    
                dist = math.sqrt((point['x'] - other_point['x'])**2 + 
                               (point['y'] - other_point['y'])**2)
                
                if dist < self.CLUSTERING_THRESHOLD:
                    cluster_points.append(other_point)
                    used[j] = True
            
            # Only consider clusters with appropriate point count
            if self.MIN_CLUSTER_POINTS <= len(cluster_points) <= self.MAX_CLUSTER_POINTS:
                cluster = self.calculate_cluster_properties(cluster_points)
                clusters.append(cluster)
        
        return clusters

    def calculate_cluster_properties(self, cluster_points):
        """Calculate comprehensive properties of a point cluster"""
        # Center calculation
        center_x = sum(p['x'] for p in cluster_points) / len(cluster_points)
        center_y = sum(p['y'] for p in cluster_points) / len(cluster_points)
        center_range = math.sqrt(center_x**2 + center_y**2)
        center_angle = math.atan2(center_y, center_x)
        
        # Calculate cluster span (physical diameter)
        max_span = 0.0
        for i, p1 in enumerate(cluster_points):
            for j, p2 in enumerate(cluster_points[i+1:], i+1):
                dist = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                max_span = max(max_span, dist)
        
        # Angular span calculation
        angles = [p['angle'] for p in cluster_points]
        angular_span_rad = max(angles) - min(angles)
        angular_span_deg = math.degrees(angular_span_rad)
        
        # Range statistics for compactness
        ranges = [p['range'] for p in cluster_points]
        range_std = np.std(ranges)
        range_mean = np.mean(ranges)
        
        return {
            'points': len(cluster_points),
            'center_x': center_x,
            'center_y': center_y,
            'center_range': center_range,
            'center_angle': center_angle,
            'span': max_span,
            'angular_span_deg': angular_span_deg,
            'range_std': range_std,
            'range_mean': range_mean,
            'raw_points': cluster_points
        }

    def analyze_as_cylinder(self, cluster):
        """Analyze cluster as potential cylinder with strict validation"""
        diameter = cluster['span']
        radius = diameter / 2.0
        distance = cluster['center_range']
        angular_span = cluster['angular_span_deg']
        
        # 1. Size validation (4-6cm diameter)
        size_valid = self.MIN_DIAMETER <= diameter <= self.MAX_DIAMETER
        size_score = 1.0 if size_valid else 0.0
        
        # 2. Point count validation (8-25 points in close range)
        point_count = cluster['points']
        point_valid = self.MIN_CLUSTER_POINTS <= point_count <= self.MAX_CLUSTER_POINTS
        point_score = 1.0 if point_valid else 0.0
        
        # 3. Angular span validation (4-8 degrees)
        span_valid = self.MIN_ANGULAR_SPAN <= angular_span <= self.MAX_ANGULAR_SPAN
        span_score = 1.0 if span_valid else 0.0
        
        # 4. Geometric consistency check
        # For cylinder: theoretical_angular_span = 2 * arcsin(radius / distance)
        if distance > 0 and radius > 0:
            theoretical_span_rad = 2 * math.asin(min(1.0, radius / distance))
            theoretical_span_deg = math.degrees(theoretical_span_rad)
            span_error = abs(angular_span - theoretical_span_deg)
            geometry_valid = span_error <= self.GEOMETRY_TOLERANCE
            geometry_score = 1.0 if geometry_valid else 0.0
        else:
            geometry_valid = False
            geometry_score = 0.0
            theoretical_span_deg = 0.0
        
        # 5. Compactness validation (low range variation)
        compactness_valid = cluster['range_std'] < 0.05  # 5cm std dev max
        compactness_score = 1.0 if compactness_valid else 0.0
        
        # All criteria must pass for high confidence
        all_criteria_passed = (size_valid and point_valid and span_valid and 
                              geometry_valid and compactness_valid)
        
        # Conservative confidence calculation
        if all_criteria_passed:
            confidence = min(0.95, (size_score + point_score + span_score + 
                                  geometry_score + compactness_score) / 5.0)
        else:
            confidence = 0.0
        
        # Final cylinder determination
        is_cylinder = confidence >= self.MIN_CONFIDENCE
        
        return {
            'is_cylinder': is_cylinder,
            'confidence': confidence,
            'distance': distance,
            'radius': radius,
            'diameter': diameter,
            'points': point_count,
            'angular_span': angular_span,
            'theoretical_span': theoretical_span_deg,
            'span_error': abs(angular_span - theoretical_span_deg) if distance > 0 else 0,
            'compactness': cluster['range_std'],
            'validation_details': {
                'size_valid': size_valid,
                'point_valid': point_valid,
                'span_valid': span_valid,
                'geometry_valid': geometry_valid,
                'compactness_valid': compactness_valid
            }
        }

    def report_cylinder_detections(self, cylinders):
        """Report all detected cylinders with comprehensive details"""
        cylinder_count = len(cylinders)
        
        if cylinder_count == 0:
            self.get_logger().info("0 cylinders detected in range")
            return
        
        # Sort cylinders by distance (closest first)
        cylinders_sorted = sorted(cylinders, key=lambda c: c['distance'])
        
        # Report count and details
        self.get_logger().info(f"{cylinder_count} cylinder{'s' if cylinder_count != 1 else ''} detected:")
        
        for i, cyl in enumerate(cylinders_sorted, 1):
            self.get_logger().info(
                f"  Cylinder {i}: distance={cyl['distance']:.2f}m, "
                f"confidence={cyl['confidence']:.2f}, "
                f"radius={cyl['radius']*100:.1f}cm, "
                f"points={cyl['points']}, "
                f"span={cyl['angular_span']:.1f}°"
            )
        
        # Identify largest cylinder
        largest_cylinder = max(cylinders_sorted, key=lambda c: c['radius'])
        self.get_logger().info(
            f"Largest cylinder: radius={largest_cylinder['radius']*100:.1f}cm "
            f"at {largest_cylinder['distance']:.2f}m"
        )
        
        # Optional: Show validation details for debugging
        self.get_logger().debug("=== Validation Details ===")
        for i, cyl in enumerate(cylinders_sorted, 1):
            details = cyl['validation_details']
            self.get_logger().debug(
                f"Cylinder {i}: Size={details['size_valid']}, "
                f"Points={details['point_valid']}, "
                f"Span={details['span_valid']}, "
                f"Geometry={details['geometry_valid']}, "
                f"Compact={details['compactness_valid']}"
            )

def main(args=None):
    rclpy.init(args=args)
    
    detector = MultiCylinderDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info("Multi-cylinder detection stopped")
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
