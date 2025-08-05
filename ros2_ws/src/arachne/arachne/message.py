'''
Node name: message.py
This code is a ROS2 node that recieves user-sent 
messages and stores the most recent 20 messages in a queue.

Inputs: User messages via 'new_message' topic
Outputs: Queued messages via 'messages' topic
'''
# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Standard library imports
import json

# Config file lol
from config import MESSAGE_QUEUE_SIZE

class MessageNode(Node):
    def __init__(self):
        super().__init__('message_node')
        
        # Initialize message queue
        self.message_queue = []
        
        # Create publisher for messages
        self.message_publisher = self.create_publisher(String, 'messages', 10)
        
        # Create subscription to new messages
        self.create_subscription(String, 'new_message', self.new_message_callback, 10)

    def new_message_callback(self, msg):
        """Handle incoming messages and maintain a queue"""
        if len(self.message_queue) >= MESSAGE_QUEUE_SIZE:
            self.message_queue.pop(0)  # Remove oldest message if queue is full
        self.message_queue.append(msg.data)
        
        # Publish the current message queue
        self.publish_messages()

    def publish_messages(self):
        """Publish the current message queue as a JSON string"""
        msg_data = json.dumps(self.message_queue)
        msg = String()
        msg.data = msg_data
        self.message_publisher.publish(msg)

def main():
    rclpy.init()
    node = MessageNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()