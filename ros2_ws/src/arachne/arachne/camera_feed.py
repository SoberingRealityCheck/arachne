'''
Node name: camera_feed.py
This code is a ROS2 node that publishes camera feed data.
Handles errors with the connection, retries, and 
potentially also compression of the data if it's too much to transmit quickly.

Also publishes a separate topic ('camera_ascii') with an ASCII art representation of the camera feed.

Inputs: Camera feed via hardware interface
Outputs: Published camera feed image to 'camera_feed' topic
'''

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from picamera2 import Picamera2, Preview
from ascii_magic import AsciiArt

class CameraFeedNode(Node):
    def __init__(self):
        super().__init__('camera_feed_node')
        self.publisher_ = self.create_publisher(Image, 'camera_feed', 10)
        self.publisher_ascii_ = self.create_publisher(String, 'camera_ascii', 10)
        self.picamera2 = Picamera2()
        self.picamera2.start_preview(Preview.NULL)
        self.timer = self.create_timer(0.1, self.publish_camera_feed)

    def ascii_ify(self, image):
        # This implementation might not work, so leaving space now for
        # something more complicated if i need it.
        ascii_image = AsciiArt.from_image(image)
        return ascii_image.to_ascii(columns = 100, monochrome = False)
    
    def publish_camera_feed(self):
        try:
            image = self.picamera2.capture_array()
            msg = Image()
            msg.height = image.shape[0]
            msg.width = image.shape[1]
            msg.encoding = 'rgb8'
            msg.data = image.tobytes()
            msg.step = image.shape[1] * 3
            self.publisher_.publish(msg)
            self.get_logger().info('Published camera feed')
            
            # For Funsies: Conversion to ascii art
            ascii_image = self.ascii_ify(image)
            self.publisher_ascii_.publish(ascii_image)
            self.get_logger().info('Published ASCII art representation of camera feed')

        except ConnectionError:
            self.get_logger().error('Connection error while capturing camera feed. Retrying...')
            self.picamera2.close()
            self.picamera2 = Picamera2()
            self.picamera2.start_preview(Preview.NULL)
        except Exception as e:
            self.get_logger().error(f'Error capturing or publishing camera feed: {e}')
    
def main():
    rclpy.init()
    node = CameraFeedNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()