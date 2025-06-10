#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import tf2_ros
import tf2_geometry_msgs

# ============== EASILY CONFIGURABLE PARAMETERS ==============
NUM_ROWS = 3          # Number of tree rows
TREES_PER_ROW = 5     # Number of trees in each row
TREE_SPACING = 0.3    # Distance between trees in a row (meters)
ROW_SPACING = 0.5     # Distance between rows (meters)

# Tree marker appearance
TREE_RADIUS = 0.05    # Radius of tree trunk (meters)
TREE_HEIGHT = 0.5     # Height of tree marker (meters)
# ==========================================================

class TreeMarkerPublisher:
    def __init__(self):
        rospy.init_node('tree_marker_publisher', anonymous=True)
        
        # Publisher for tree positions
        self.marker_pub = rospy.Publisher('/tree_pos', MarkerArray, queue_size=10)
        
        # TF2 buffer and listener for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Wait a moment for transforms to become available
        rospy.sleep(1.0)
        
        # Generate markers once at startup
        self.marker_array = self.generate_tree_markers()
        
        # Publishing rate
        self.rate = rospy.Rate(1)  # 1 Hz
        
        rospy.loginfo(f"Tree marker publisher initialized with {NUM_ROWS} rows and {TREES_PER_ROW} trees per row")
        
    def generate_tree_markers(self):
        """Generate tree markers in a grid pattern relative to robot's initial position"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # Get robot's current position in odom frame
        try:
            # Try to get transform from base_link to odom
            transform = self.tf_buffer.lookup_transform('odom', 'base_link', rospy.Time())
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            rospy.loginfo(f"Robot initial position: x={robot_x:.2f}, y={robot_y:.2f}")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logwarn("Could not get robot transform, using (0,0) as origin")
            robot_x = 0.0
            robot_y = 0.0
        
        # Generate markers for each tree
        for row in range(NUM_ROWS):
            for tree in range(TREES_PER_ROW):
                marker = Marker()
                
                # Basic marker properties
                marker.header.frame_id = "odom"  # Fixed frame so markers don't move
                marker.header.stamp = rospy.Time.now()
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
                marker.lifetime = rospy.Duration()
                
                marker_array.markers.append(marker)
                marker_id += 1
                
        rospy.loginfo(f"Generated {marker_id} tree markers")
        return marker_array
    
    def publish_markers(self):
        """Publish the marker array"""
        # Update timestamp for all markers
        for marker in self.marker_array.markers:
            marker.header.stamp = rospy.Time.now()
        
        self.marker_pub.publish(self.marker_array)
    
    def run(self):
        """Main loop"""
        rospy.loginfo("Starting to publish tree markers...")
        
        while not rospy.is_shutdown():
            self.publish_markers()
            self.rate.sleep()

def main():
    try:
        node = TreeMarkerPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()