#!/usr/bin/env python3
"""
Cylinder Feature Analyzer for ROSbot 3 Pro
Analyzes LiDAR data to characterize cylinder detection properties
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import numpy as np
import math
import json
import time
from collections import defaultdict

class CylinderAnalyzer(Node):
    def __init__(self):
        super().__init__('cylinder_analyzer')
        
        # Subscriber
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        # Publisher for analysis results
        self.analysis_pub = self.create_publisher(String, '/cylinder_analysis', 10)
        
        # Analysis parameters
        self.MIN_RANGE = 0.05  # 5cm minimum
        self.MAX_RANGE = 2.0   # 2m maximum
        self.CLUSTERING_THRESHOLD = 0.05  # 5cm clustering
        self.WALL_DISTANCE_THRESHOLD = 0.3  # Assume walls are >30cm clusters
        
        # Data collection
        self.scan_data = []
        self.analysis_results = []
        self.scan_count = 0
        self.collection_duration = 10.0  # Collect data for 10 seconds
        self.start_time = time.time()
        
        # Analysis timer
        self.analysis_timer = self.create_timer(1.0, self.periodic_analysis)
        
        self.get_logger().info("Cylinder Analyzer started")
        self.get_logger().info("Place cylinder in front of robot and wait for analysis...")
        self.get_logger().info(f"Collecting data for {self.collection_duration} seconds...")

    def scan_callback(self, msg):
        """Collect and analyze LiDAR scan data"""
        if time.time() - self.start_time > self.collection_duration:
            return  # Stop collecting after duration
            
        self.scan_count += 1
        
        # Process current scan
        analysis = self.analyze_single_scan(msg)
        if analysis:
            self.scan_data.append({
                'scan_id': self.scan_count,
                'timestamp': time.time(),
                'raw_ranges': list(msg.ranges),
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'analysis': analysis
            })
            
        # Log progress
        if self.scan_count % 10 == 0:
            self.get_logger().info(f"Processed {self.scan_count} scans...")

    def analyze_single_scan(self, msg):
        """Analyze a single LiDAR scan for cylinder characteristics"""
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Remove invalid readings
        valid_mask = np.isfinite(ranges) & (ranges >= self.MIN_RANGE) & (ranges <= self.MAX_RANGE)
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]
        
        if len(valid_ranges) == 0:
            return None
        
        # Convert to cartesian coordinates
        points = []
        for r, a in zip(valid_ranges, valid_angles):
            x = r * math.cos(a)
            y = r * math.sin(a)
            points.append({'x': x, 'y': y, 'range': r, 'angle': a})
        
        # Cluster points
        clusters = self.cluster_points(points)
        
        # Classify clusters
        walls = []
        cylinder_candidates = []
        
        for cluster in clusters:
            if cluster['span'] > self.WALL_DISTANCE_THRESHOLD:
                walls.append(cluster)
            else:
                cylinder_candidates.append(cluster)
        
        # Analyze cylinder candidates
        cylinder_analysis = []
        for candidate in cylinder_candidates:
            analysis = self.analyze_cluster_as_cylinder(candidate)
            cylinder_analysis.append(analysis)
        
        return {
            'total_points': len(points),
            'total_clusters': len(clusters),
            'wall_clusters': len(walls),
            'cylinder_candidates': len(cylinder_candidates),
            'cylinder_analysis': cylinder_analysis,
            'walls': walls,
            'all_clusters': clusters
        }

    def cluster_points(self, points):
        """Cluster LiDAR points using simple distance-based clustering"""
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
            
            # Calculate cluster properties
            cluster = self.calculate_cluster_properties(cluster_points)
            clusters.append(cluster)
        
        return clusters

    def calculate_cluster_properties(self, cluster_points):
        """Calculate comprehensive properties of a point cluster"""
        # Basic statistics
        center_x = sum(p['x'] for p in cluster_points) / len(cluster_points)
        center_y = sum(p['y'] for p in cluster_points) / len(cluster_points)
        center_range = math.sqrt(center_x**2 + center_y**2)
        center_angle = math.atan2(center_y, center_x)
        
        # Cluster span (maximum distance between any two points)
        max_span = 0.0
        min_span = float('inf')
        for i, p1 in enumerate(cluster_points):
            for j, p2 in enumerate(cluster_points[i+1:], i+1):
                dist = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
                max_span = max(max_span, dist)
                min_span = min(min_span, dist) if dist > 0 else min_span
        
        # Angular span
        angles = [p['angle'] for p in cluster_points]
        angular_span = max(angles) - min(angles)
        
        # Range statistics
        ranges = [p['range'] for p in cluster_points]
        range_mean = np.mean(ranges)
        range_std = np.std(ranges)
        range_min = min(ranges)
        range_max = max(ranges)
        
        # Density (points per unit angle)
        density = len(cluster_points) / max(angular_span, 0.001)  # Avoid division by zero
        
        return {
            'points': len(cluster_points),
            'center_x': center_x,
            'center_y': center_y,
            'center_range': center_range,
            'center_angle': center_angle,
            'center_angle_deg': math.degrees(center_angle),
            'span': max_span,
            'min_span': min_span if min_span != float('inf') else 0.0,
            'angular_span': angular_span,
            'angular_span_deg': math.degrees(angular_span),
            'range_mean': range_mean,
            'range_std': range_std,
            'range_min': range_min,
            'range_max': range_max,
            'density': density,
            'raw_points': cluster_points
        }

    def analyze_cluster_as_cylinder(self, cluster):
        """Analyze if a cluster could be a cylinder and extract its properties"""
        # Cylinder-specific metrics
        aspect_ratio = cluster['span'] / max(cluster['center_range'] * cluster['angular_span'], 0.001)
        
        # Compactness (how circular the point distribution is)
        center_x, center_y = cluster['center_x'], cluster['center_y']
        distances_from_center = []
        for point in cluster['raw_points']:
            dist = math.sqrt((point['x'] - center_x)**2 + (point['y'] - center_y)**2)
            distances_from_center.append(dist)
        
        compactness = np.std(distances_from_center) / max(np.mean(distances_from_center), 0.001)
        
        # Estimate cylinder diameter based on span
        estimated_diameter = cluster['span']
        
        # Confidence scoring (0-1)
        # Higher confidence for:
        # - 2+ points
        # - Low compactness (circular)
        # - Reasonable size (2-10cm span)
        # - High density
        
        confidence = 0.0
        if cluster['points'] >= 2:
            confidence += 0.3
        if cluster['points'] >= 3:
            confidence += 0.2
        if 0.02 <= cluster['span'] <= 0.10:  # 2-10cm span
            confidence += 0.3
        if compactness < 0.5:  # Compact cluster
            confidence += 0.1
        if cluster['density'] > 10:  # Dense cluster
            confidence += 0.1
        
        confidence = min(confidence, 1.0)
        
        return {
            'cluster_properties': cluster,
            'estimated_diameter': estimated_diameter,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'confidence': confidence,
            'classification': 'HIGH_CONFIDENCE_CYLINDER' if confidence > 0.7 else
                           'MEDIUM_CONFIDENCE_CYLINDER' if confidence > 0.4 else
                           'LOW_CONFIDENCE_CYLINDER'
        }

    def periodic_analysis(self):
        """Perform periodic analysis and publish results"""
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > self.collection_duration:
            # Final analysis
            self.final_analysis()
            self.analysis_timer.cancel()
            return
        
        # Progress update
        remaining = self.collection_duration - elapsed_time
        self.get_logger().info(f"Data collection: {elapsed_time:.1f}s / {self.collection_duration}s "
                              f"({self.scan_count} scans, {remaining:.1f}s remaining)")

    def final_analysis(self):
        """Perform comprehensive analysis of all collected data"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("FINAL CYLINDER ANALYSIS RESULTS")
        self.get_logger().info("=" * 60)
        
        if not self.scan_data:
            self.get_logger().error("No scan data collected!")
            return
        
        # Aggregate statistics
        all_cylinder_candidates = []
        confidence_scores = []
        diameters = []
        distances = []
        point_counts = []
        angular_spans = []
        
        for scan in self.scan_data:
            for cylinder in scan['analysis']['cylinder_analysis']:
                all_cylinder_candidates.append(cylinder)
                confidence_scores.append(cylinder['confidence'])
                diameters.append(cylinder['estimated_diameter'])
                distances.append(cylinder['cluster_properties']['center_range'])
                point_counts.append(cylinder['cluster_properties']['points'])
                angular_spans.append(cylinder['cluster_properties']['angular_span_deg'])
        
        # Summary statistics
        self.get_logger().info(f"Total scans processed: {len(self.scan_data)}")
        self.get_logger().info(f"Total cylinder candidates found: {len(all_cylinder_candidates)}")
        
        if confidence_scores:
            self.get_logger().info(f"Average confidence: {np.mean(confidence_scores):.3f}")
            self.get_logger().info(f"Max confidence: {np.max(confidence_scores):.3f}")
            self.get_logger().info(f"Min confidence: {np.min(confidence_scores):.3f}")
            self.get_logger().info(f"Confidence std dev: {np.std(confidence_scores):.3f}")
        
        if diameters:
            self.get_logger().info(f"Estimated diameter - Mean: {np.mean(diameters)*100:.2f}cm")
            self.get_logger().info(f"Estimated diameter - Std: {np.std(diameters)*100:.2f}cm")
            self.get_logger().info(f"Estimated diameter - Min: {np.min(diameters)*100:.2f}cm")
            self.get_logger().info(f"Estimated diameter - Max: {np.max(diameters)*100:.2f}cm")
        
        if distances:
            self.get_logger().info(f"Detection distance - Mean: {np.mean(distances):.3f}m")
            self.get_logger().info(f"Detection distance - Min: {np.min(distances):.3f}m")
            self.get_logger().info(f"Detection distance - Max: {np.max(distances):.3f}m")
            self.get_logger().info(f"Detection distance - Std: {np.std(distances):.3f}m")
        
        if point_counts:
            self.get_logger().info(f"Points per detection - Mean: {np.mean(point_counts):.2f}")
            self.get_logger().info(f"Points per detection - Min: {int(np.min(point_counts))}")
            self.get_logger().info(f"Points per detection - Max: {int(np.max(point_counts))}")
            self.get_logger().info(f"Points per detection - Mode: {int(np.bincount(point_counts).argmax()) if point_counts else 'N/A'}")
        
        if angular_spans:
            self.get_logger().info(f"Angular span - Mean: {np.mean(angular_spans):.2f}°")
            self.get_logger().info(f"Angular span - Min: {np.min(angular_spans):.2f}°")
            self.get_logger().info(f"Angular span - Max: {np.max(angular_spans):.2f}°")
            self.get_logger().info(f"Angular span - Std: {np.std(angular_spans):.2f}°")
        
        # Detection quality analysis
        high_confidence = [c for c in all_cylinder_candidates if c['confidence'] > 0.7]
        medium_confidence = [c for c in all_cylinder_candidates if 0.4 <= c['confidence'] <= 0.7]
        low_confidence = [c for c in all_cylinder_candidates if c['confidence'] < 0.4]
        
        self.get_logger().info("=" * 40)
        self.get_logger().info("DETECTION QUALITY BREAKDOWN:")
        self.get_logger().info(f"High confidence detections (>0.7): {len(high_confidence)}")
        self.get_logger().info(f"Medium confidence detections (0.4-0.7): {len(medium_confidence)}")
        self.get_logger().info(f"Low confidence detections (<0.4): {len(low_confidence)}")
        
        # Distance-based analysis
        if distances:
            close_detections = [c for c in all_cylinder_candidates if c['cluster_properties']['center_range'] < 0.5]
            medium_detections = [c for c in all_cylinder_candidates if 0.5 <= c['cluster_properties']['center_range'] < 1.0]
            far_detections = [c for c in all_cylinder_candidates if c['cluster_properties']['center_range'] >= 1.0]
            
            self.get_logger().info("=" * 40)
            self.get_logger().info("DISTANCE-BASED ANALYSIS:")
            self.get_logger().info(f"Close range (<0.5m): {len(close_detections)} detections")
            if close_detections:
                close_conf = [c['confidence'] for c in close_detections]
                close_points = [c['cluster_properties']['points'] for c in close_detections]
                self.get_logger().info(f"  Avg confidence: {np.mean(close_conf):.3f}")
                self.get_logger().info(f"  Avg points: {np.mean(close_points):.1f}")
            
            self.get_logger().info(f"Medium range (0.5-1.0m): {len(medium_detections)} detections")
            if medium_detections:
                med_conf = [c['confidence'] for c in medium_detections]
                med_points = [c['cluster_properties']['points'] for c in medium_detections]
                self.get_logger().info(f"  Avg confidence: {np.mean(med_conf):.3f}")
                self.get_logger().info(f"  Avg points: {np.mean(med_points):.1f}")
            
            self.get_logger().info(f"Far range (>1.0m): {len(far_detections)} detections")
            if far_detections:
                far_conf = [c['confidence'] for c in far_detections]
                far_points = [c['cluster_properties']['points'] for c in far_detections]
                self.get_logger().info(f"  Avg confidence: {np.mean(far_conf):.3f}")
                self.get_logger().info(f"  Avg points: {np.mean(far_points):.1f}")
        
        # Best detection examples
        if confidence_scores:
            best_idx = np.argmax(confidence_scores)
            best_detection = all_cylinder_candidates[best_idx]
            
            self.get_logger().info("=" * 40)
            self.get_logger().info("BEST DETECTION EXAMPLE:")
            self.get_logger().info(f"  Confidence: {best_detection['confidence']:.3f}")
            self.get_logger().info(f"  Estimated diameter: {best_detection['estimated_diameter']*100:.2f}cm")
            self.get_logger().info(f"  Distance: {best_detection['cluster_properties']['center_range']:.3f}m")
            self.get_logger().info(f"  LiDAR points: {best_detection['cluster_properties']['points']}")
            self.get_logger().info(f"  Angular span: {best_detection['cluster_properties']['angular_span_deg']:.2f}°")
            self.get_logger().info(f"  Physical span: {best_detection['cluster_properties']['span']*100:.2f}cm")
            self.get_logger().info(f"  Compactness: {best_detection['compactness']:.3f}")
            self.get_logger().info(f"  Classification: {best_detection['classification']}")
        
        # Detection frequency analysis
        detection_rate = len(all_cylinder_candidates) / len(self.scan_data) * 100
        self.get_logger().info("=" * 40)
        self.get_logger().info("DETECTION STATISTICS:")
        self.get_logger().info(f"Detection rate: {detection_rate:.1f}% of scans")
        self.get_logger().info(f"Scans with no detection: {len(self.scan_data) - len([s for s in self.scan_data if s['analysis']['cylinder_candidates'] > 0])}")
        self.get_logger().info(f"Scans with 1+ detections: {len([s for s in self.scan_data if s['analysis']['cylinder_candidates'] > 0])}")
        
        # Raw data summary for copying
        self.get_logger().info("=" * 60)
        self.get_logger().info("RAW DATA SUMMARY (for copy-paste):")
        self.get_logger().info("=" * 60)
        
        summary_data = {
            "total_scans": len(self.scan_data),
            "total_detections": len(all_cylinder_candidates),
            "detection_rate_percent": round(detection_rate, 1),
            "confidence_stats": {
                "mean": round(np.mean(confidence_scores), 3) if confidence_scores else 0,
                "std": round(np.std(confidence_scores), 3) if confidence_scores else 0,
                "min": round(np.min(confidence_scores), 3) if confidence_scores else 0,
                "max": round(np.max(confidence_scores), 3) if confidence_scores else 0
            },
            "diameter_stats_cm": {
                "mean": round(np.mean(diameters)*100, 2) if diameters else 0,
                "std": round(np.std(diameters)*100, 2) if diameters else 0,
                "min": round(np.min(diameters)*100, 2) if diameters else 0,
                "max": round(np.max(diameters)*100, 2) if diameters else 0
            },
            "distance_stats_m": {
                "mean": round(np.mean(distances), 3) if distances else 0,
                "std": round(np.std(distances), 3) if distances else 0,
                "min": round(np.min(distances), 3) if distances else 0,
                "max": round(np.max(distances), 3) if distances else 0
            },
            "points_per_detection": {
                "mean": round(np.mean(point_counts), 2) if point_counts else 0,
                "min": int(np.min(point_counts)) if point_counts else 0,
                "max": int(np.max(point_counts)) if point_counts else 0,
                "mode": int(np.bincount(point_counts).argmax()) if point_counts else 0
            },
            "angular_span_degrees": {
                "mean": round(np.mean(angular_spans), 2) if angular_spans else 0,
                "std": round(np.std(angular_spans), 2) if angular_spans else 0,
                "min": round(np.min(angular_spans), 2) if angular_spans else 0,
                "max": round(np.max(angular_spans), 2) if angular_spans else 0
            },
            "quality_breakdown": {
                "high_confidence_count": len(high_confidence),
                "medium_confidence_count": len(medium_confidence),
                "low_confidence_count": len(low_confidence)
            },
            "distance_breakdown": {
                "close_range_detections": len([c for c in all_cylinder_candidates if c['cluster_properties']['center_range'] < 0.5]),
                "medium_range_detections": len([c for c in all_cylinder_candidates if 0.5 <= c['cluster_properties']['center_range'] < 1.0]),
                "far_range_detections": len([c for c in all_cylinder_candidates if c['cluster_properties']['center_range'] >= 1.0])
            }
        }
        
        # Print formatted JSON for easy copying
        import json
        self.get_logger().info("JSON_START")
        print(json.dumps(summary_data, indent=2))
        self.get_logger().info("JSON_END")
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("ANALYSIS COMPLETE!")
        self.get_logger().info("Copy the data between JSON_START and JSON_END markers")
        self.get_logger().info("=" * 60)

def main(args=None):
    rclpy.init(args=args)
    
    analyzer = CylinderAnalyzer()
    
    try:
        rclpy.spin(analyzer)
    except KeyboardInterrupt:
        analyzer.get_logger().info("Analysis interrupted by user")
    finally:
        analyzer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
