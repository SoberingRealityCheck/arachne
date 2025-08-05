'''
Node name: state.py
This code is a ROS2 node that collects the total 
state of the robot. It aggregates data from various
sensors and components to maintain a comprehensive 
view of the robot's status.

It also serves as a state machine, ensuring that the 
robot locomotion operates in a consistent state and 
handles transitions between different states.

Inputs: 
 - IMU data via 'imu_data' topic
 - Camera feed data via 'camera_feed' topic
 - Remote movement requests via 'move_request' topic
 - Messages via 'messages' topic

Outputs:
 - Aggregated robot state via 'robot_state' topic
 - Processed movement commands via 'move_cmd' topic
'''
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import json

class RobotStateNode(Node):
    def __init__(self):
        super().__init__('robot_state_node')
        
        # Initialize robot state
        self.robot_state = {
            'imu_data': None,
            'camera_feed': None,
            'move_request': None,
            'messages': []
        }
        
        # Create publishers and subscribers
        self.state_publisher = self.create_publisher(String, 'robot_state', 10)
        self.create_subscription(String, 'imu_data', self.imu_callback, 10)
        self.create_subscription(String, 'camera_feed', self.camera_callback, 10)
        self.create_subscription(String, 'move_request', self.move_request_callback, 10)
        self.create_subscription(String, 'messages', self.messages_callback, 10)

    def imu_callback(self, msg):
        """Handle incoming IMU data"""
        self.robot_state['imu_data'] = json.loads(msg.data)
        self.publish_robot_state()

    def camera_callback(self, msg):
        """Handle incoming camera feed data"""
        self.robot_state['camera_feed'] = json.loads(msg.data)
        self.publish_robot_state()

    def move_request_callback(self, msg):
        """Handle movement requests"""
        self.robot_state['move_request'] = json.loads(msg.data)
        self.publish_robot_state()

    def messages_callback(self, msg):
        """Handle incoming messages"""
        message_data = json.loads(msg.data)
        self.robot_state['messages'].append(message_data)
        self.publish_robot_state()

    def publish_robot_state(self):
        """Publish the current robot state"""
        state_msg = String()
        state_msg.data = json.dumps(self.robot_state)
        self.state_publisher.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStateNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()