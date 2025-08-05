#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from flask import Flask, jsonify, request, abort
from flask_cors import CORS, cross_origin
import json
import os
import time
from threading import Thread
from std_msgs.msg import String

class TunnelAPINode(Node):
    def __init__(self):
        super().__init__('tunnel_api_node')
        
        # Robot state storage
        self.robot_state = {}
        
        # Subscribe to robot state updates
        self.state_subscription = self.create_subscription(
            String,
            'robot_state',
            self.robot_state_callback,
            10
        )
        
        # Initialize Flask app
        self.app = Flask(__name__)
        
        # Configure CORS
        CORS(self.app, 
             origins=["*"],  # Allow all origins
             methods=["GET", "POST", "OPTIONS"],  # Allow specific methods
             supports_credentials=True
        )
        
        # Setup routes
        self.setup_routes()
        
        # Start Flask
        self.flask_thread = Thread(target=self.run_flask_app)
        self.flask_thread.daemon = True
        self.flask_thread.start()
        
        self.get_logger().info('Tunnel API Node started with CORS enabled')

    def robot_state_callback(self, msg):
        """Update robot state from ROS2 topic"""
        try:
            self.robot_state = json.loads(msg.data)
            self.robot_state['timestamp'] = time.time()
        except json.JSONDecodeError:
            self.get_logger().error('Failed to parse robot state JSON')

    def setup_routes(self):

        @self.app.route('/api/robot-state', methods=['GET', 'OPTIONS'])
        @cross_origin()
        def get_robot_state():
            """Get current robot state"""
            if not self.robot_state:
                return jsonify({'error': 'No robot state available'}), 404
            
            return jsonify({
                'status': 'success',
                'data': self.robot_state,
                'retrieved_at': time.time()
            })

        @self.app.route('/api/health', methods=['GET', 'OPTIONS'])
        @cross_origin()
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'tunnel_mode': True,
                'cors_enabled': True
            })

    def run_flask_app(self):
        """Run Flask app on all interfaces for Docker"""
        self.app.run(
            host='0.0.0.0',
            port=int(os.getenv('API_PORT', '8080')),
            debug=False
        )

def main(args=None):
    rclpy.init(args=args)
    node = TunnelAPINode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()