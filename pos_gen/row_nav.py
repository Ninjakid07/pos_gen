import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.timer import Timer
from nav2_msgs.action import NavigateToPose
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus
import math

# ============== Navigation Parameters ===================
# This distance must be less than the REVEAL_DISTANCE in your
# other node to ensure the next tree is revealed when the robot arrives.
TARGET_DISTANCE_FROM_ROW = 0.2 # Perpendicular distance from the tree row (meters)

# The time to wait after reaching a waypoint before assuming the row has ended.
END_OF_ROW_TIMEOUT_SEC = 5.0


class ExploratoryGoalSender(Node):
    """
    Navigates a dynamically appearing row of trees with an unknown length.

    This node subscribes to a MarkerArray topic that reveals trees one by one.
    As new trees are detected, it cancels the current navigation goal and sends
    a new, updated goal to the furthest visible tree. If it reaches the end of
    the known row, it waits for a timeout period. If no new trees appear, it
    concludes the mission is complete.
    """

    def __init__(self):
        """Initializes the node, action client, and subscriber."""
        super().__init__('exploratory_goal_sender')

        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.tree_subscriber = self.create_subscription(
            MarkerArray,
            '/tree_pos',
            self.tree_marker_callback,
            10
        )

        # State tracking variables
        self.current_goal_handle = None
        self.current_goal_tree_id = -1
        self.end_of_row_timer: Timer = None

        self.get_logger().info('Exploratory Goal Sender has started.')
        self.get_logger().info(f'Will stop after a {END_OF_ROW_TIMEOUT_SEC}s timeout at the end of the row.')

    def tree_marker_callback(self, msg: MarkerArray):
        """Callback triggered when new tree marker data is received."""
        if not msg.markers:
            return

        latest_tree_id = max(marker.id for marker in msg.markers)

        if latest_tree_id > self.current_goal_tree_id:
            self.get_logger().info(f'New target detected! Farthest tree is now ID {latest_tree_id}.')
            
            # If the end-of-row timer is running, a new tree has appeared. Cancel the timer.
            if self.end_of_row_timer is not None:
                self.get_logger().info('New tree found. Canceling end-of-row timeout.')
                self.end_of_row_timer.cancel()
                self.end_of_row_timer = None

            self.current_goal_tree_id = latest_tree_id
            latest_marker = next(m for m in msg.markers if m.id == latest_tree_id)
            self.send_new_waypoint(latest_marker.pose.position)

    def send_new_waypoint(self, tree_position):
        """Cancels any old goal and sends a new one."""
        if self.current_goal_handle is not None:
            self.get_logger().info('Canceling previous goal to send a new one.')
            self.current_goal_handle.cancel_goal_async()

        goal_x = tree_position.x
        goal_y = tree_position.y + TARGET_DISTANCE_FROM_ROW
        
        self.get_logger().info(f'Calculating new waypoint for tree {self.current_goal_tree_id}: X={goal_x:.2f}, Y={goal_y:.2f}')

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = goal_x
        goal_msg.pose.pose.position.y = goal_y
        goal_msg.pose.pose.orientation.w = 1.0

        self.get_logger().info('Sending new goal to Nav2...')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle the server's response to sending a goal."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by Nav2 server.')
            return

        self.current_goal_handle = goal_handle
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle the final result of a navigation goal."""
        status = future.result().status
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Successfully reached waypoint for tree ID {self.current_goal_tree_id}.')
            self.get_logger().info(f'Starting {END_OF_ROW_TIMEOUT_SEC}s timer to wait for next tree...')
            
            # Start the timer to check if the row is truly over.
            self.end_of_row_timer = self.create_timer(END_OF_ROW_TIMEOUT_SEC, self.declare_mission_complete)

        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info('Goal was canceled, likely by a new goal being sent.')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('Navigation was aborted by Nav2. Shutting down.')
            rclpy.shutdown()

    def declare_mission_complete(self):
        """Called by the timer if no new trees are found in time."""
        self.get_logger().info('End-of-row timeout expired. No new trees detected.')
        self.get_logger().info('Mission Complete!')
        
        # Cleanly stop the timer and shut down
        self.end_of_row_timer.cancel()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    
    exploratory_goal_sender = ExploratoryGoalSender()
    
    exploratory_goal_sender.get_logger().info('Waiting for Nav2 action server (/navigate_to_pose)...')
    exploratory_goal_sender._action_client.wait_for_server()
    exploratory_goal_sender.get_logger().info('Action server is available.')

    try:
        rclpy.spin(exploratory_goal_sender)
    except KeyboardInterrupt:
        exploratory_goal_sender.get_logger().info('Execution interrupted by user.')
    finally:
        if rclpy.ok():
            exploratory_goal_sender.destroy_node()
            # The shutdown is now handled by the timer or callbacks
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
