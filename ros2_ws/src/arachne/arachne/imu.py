'''
Node name: imu.py
This code is a ROS2 node that processes IMU data.

Inputs: raw IMU data via hardware interface
Outputs: Processed position / rotation / velocity data to 'imu_data' topic
'''
import time

# ROS2 Imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Hardware Library
import sys
import os

# Hard-code the absolute path to the 'lib' directory and add it to sys.path
LIB_PATH = '/ros2/ros2_ws/src/arachne/arachne/lib'  # Replace with the actual absolute path
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

from icm20948 import ICM20948

class IMU(Node):
    
    def __init__(self):
        super().__init__('imu_node')
        self.imu = ICM20948()
        self.publisher = self.create_publisher(String, 'imu_data', 10)
        self.timer = self.create_timer(0.25, self.publish_imu_data)
        self.get_logger().info('IMU Node started')

    def publish_imu_data(self):
        x, y, z = self.imu.read_magnetometer_data()
        ax, ay, az, gx, gy, gz = self.imu.read_accelerometer_gyro_data()
        
        msg = String()
        msg.data = f"Accel: {ax:.2f} {ay:.2f} {az:.2f}, Gyro: {gx:.2f} {gy:.2f} {gz:.2f}, Mag: {x:.2f} {y:.2f} {z:.2f}"
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published IMU data: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = IMU()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()