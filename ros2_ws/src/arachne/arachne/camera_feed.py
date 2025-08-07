'''
Node name: camera_feed.py
This code is a ROS2 node that publishes camera feed data.
Handles errors with the connection, retries, and 
potentially also compression of the data if it's too much to transmit quickly.

Also publishes a separate topic ('camera_ascii') with an ASCII art representation of the camera feed.

Inputs: Camera feed via hardware interface
Outputs: Published camera feed image to 'camera_feed' topic
'''
import os
import sys

# Set environment for headless operation
os.environ['LIBCAMERA_LOG_LEVELS'] = 'ERROR'
os.environ['DISPLAY'] = ':99'

# Mock the KMS modules BEFORE importing picamera2 to trick it into thinking we have 
# the rpi hardware packages muahahahaha dumbass program didnt think we could lie to it 
class MockKMSModule:
    def __getattr__(self, name):
        class MockObject:
            def __init__(self, *args, **kwargs):
                pass
            def __call__(self, *args, **kwargs):
                return self
            def __getattr__(self, name):
                return MockObject()
        return MockObject()

sys.modules['pykms'] = MockKMSModule()
sys.modules['kms'] = MockKMSModule()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# Now we can safely import picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError as e:
    print(f"Picamera2 not available: {e}")
    PICAMERA2_AVAILABLE = False

# Import OpenCV as fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"OpenCV not available: {e}")
    CV2_AVAILABLE = False

from ascii_magic import AsciiArt
import ascii_magic
import numpy as np
import time 



class CameraFeedNode(Node):
    def __init__(self):
        super().__init__('camera_feed_node')
        self.publisher_ = self.create_publisher(Image, 'camera_feed', 10)
        self.publisher_ascii_ = self.create_publisher(Image, 'camera_ascii', 10)
        
        # Camera instance will be set by initialization
        self.picamera2 = None
        self.cv2_capture = None
        self.camera_type = None  # 'picamera2' or 'cv2'
        
        # Cooldown for warning messages to avoid log spam
        self.last_warning_time = 0
        self.warning_cooldown = 5.0  # seconds
        
        # Performance optimizations
        self.ascii_frame_skip = 5  # Only process ASCII every 5th frame
        self.frame_counter = 0
        
        # Debug camera detection
        self.initialize_camera()
        # Increase FPS - 30 FPS target (0.033 seconds)
        self.timer = self.create_timer(0.033, self.publish_camera_feed)

    def initialize_camera(self):
        """Try to initialize camera with Picamera2 first, then fall back to OpenCV/V4L2."""
        try:
            self.get_logger().info('Starting camera detection process...')
            
            # Try Picamera2 first
            if PICAMERA2_AVAILABLE and self.try_picamera2():
                self.camera_type = 'picamera2'
                self.get_logger().info('Successfully initialized camera with Picamera2')
                return
            
            # Try to configure the media pipeline before OpenCV
            self.configure_media_pipeline()
            
            # Fall back to OpenCV/V4L2
            if CV2_AVAILABLE and self.try_opencv():
                self.camera_type = 'cv2'
                self.get_logger().info('Successfully initialized camera with OpenCV/V4L2')
                return
                
            # No camera could be initialized
            self.get_logger().error('Failed to initialize any camera interface')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize camera: {e}')
            import traceback
            self.get_logger().error(f'Overall camera init traceback: {traceback.format_exc()}')

    def configure_media_pipeline(self):
        """Try to configure the media pipeline to enable camera streaming."""
        try:
            import subprocess
            self.get_logger().info('Attempting to configure media pipeline...')
            
            # Configure the pipeline with the working commands we discovered
            commands = [
                # First, set up the sensor to CSI2 link (this was the missing piece)
                ['media-ctl', '--device', '/dev/media0', '--links', '"imx708 16-001a":0->"csi2":0[1]'],
                
                # Set CSI2 format (this worked in our tests)
                ['media-ctl', '--device', '/dev/media0', '--set-v4l2', '"csi2":4[fmt:SRGGB10_1X10/640x480]'],
                
                # Enable the link from CSI2 to video0 (raw bayer path)
                ['media-ctl', '--device', '/dev/media0', '--links', '"csi2":4->"rp1-cfe-csi2_ch0":0[1]'],
                
                # Also enable the pisp-fe path for processed images on video4
                ['media-ctl', '--device', '/dev/media0', '--links', '"csi2":4->"pisp-fe":0[1]'],
                ['media-ctl', '--device', '/dev/media0', '--links', '"pisp-fe":2->"rp1-cfe-fe_image0":0[1]'],
                
                # Set video device formats
                ['v4l2-ctl', '--device', '/dev/video0', '--set-fmt-video=width=640,height=480,pixelformat=RGGB'],
                ['v4l2-ctl', '--device', '/dev/video4', '--set-fmt-video=width=640,height=480,pixelformat=BGR3'],
            ]
            
            for cmd in commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        self.get_logger().info(f'Successfully ran: {" ".join(cmd)}')
                    else:
                        self.get_logger().warn(f'Command failed: {" ".join(cmd)}, stderr: {result.stderr}')
                except Exception as e:
                    self.get_logger().warn(f'Could not run {" ".join(cmd)}: {e}')
                    
        except Exception as e:
            self.get_logger().warn(f'Media pipeline configuration failed: {e}')

    def try_picamera2(self):
        """Try to initialize camera using Picamera2."""
        try:
            self.get_logger().info('Trying Picamera2...')
            
            # First, let's check what cameras are available globally
            try:
                cameras = Picamera2.global_camera_info()
                self.get_logger().info(f'global_camera_info returned: {cameras}')
                self.get_logger().info(f'Number of cameras found: {len(cameras)}')
                
                if len(cameras) == 0:
                    self.get_logger().warn('No cameras found by Picamera2.global_camera_info(), falling back to OpenCV')
                    return False
                
                for i, cam in enumerate(cameras):
                    self.get_logger().info(f'Camera {i}: {cam}')
                    
            except Exception as e:
                self.get_logger().error(f'Could not get global camera info: {e}')
                return False
                
            # Try to create with specific camera index
            try:
                self.get_logger().info('Attempting to create Picamera2 with camera_num=0...')
                self.picamera2 = Picamera2(camera_num=0)
                self.get_logger().info('Created Picamera2 with camera_num=0')
            except Exception as e:
                self.get_logger().error(f'Failed with camera_num=0: {e}')
                return False
            
            # Configure and start camera
            try:
                camera_props = self.picamera2.camera_properties
                self.get_logger().info(f'Camera properties: {camera_props}')
                
                self.get_logger().info('Configuring camera for performance...')
                # Use video configuration instead of still for better FPS
                # Configure for 1920x1080 to avoid resizing later
                video_config = self.picamera2.create_video_configuration(
                    main={"size": (1920, 1080), "format": "RGB888"}
                )
                self.picamera2.configure(video_config)
                self.get_logger().info('Starting camera...')
                self.picamera2.start()
                self.get_logger().info('Picamera2 started successfully with video configuration!')
                return True
                
            except Exception as e:
                self.get_logger().error(f'Could not configure/start Picamera2: {e}')
                try:
                    self.picamera2.close()
                except:
                    pass
                self.picamera2 = None
                return False
                
        except Exception as e:
            self.get_logger().error(f'Picamera2 initialization failed: {e}')
            return False

    def try_opencv(self):
        """Try to initialize camera using OpenCV with V4L2."""
        try:
            self.get_logger().info('Trying OpenCV with V4L2...')
            
            # Try more video devices (we saw video0-7 in media-ctl output)
            for device_id in [0, 1, 2, 3, 4, 5, 6, 7]:
                try:
                    self.get_logger().info(f'Trying /dev/video{device_id}...')
                    cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
                    
                    if cap.isOpened():
                        # Try to set some format first
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Test if we can read a frame
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.get_logger().info(f'Successfully opened /dev/video{device_id} and captured frame')
                            self.cv2_capture = cap
                            return True
                        else:
                            self.get_logger().warn(f'/dev/video{device_id} opened but cannot read frames (ret={ret}, frame={frame is not None})')
                            
                            # Try different format
                            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                self.get_logger().info(f'Successfully opened /dev/video{device_id} with YUYV format')
                                self.cv2_capture = cap
                                return True
                            
                            cap.release()
                    else:
                        self.get_logger().warn(f'Could not open /dev/video{device_id}')
                        
                except Exception as e:
                    self.get_logger().warn(f'Error trying /dev/video{device_id}: {e}')
                    
            # Also try without specifying V4L2 backend
            self.get_logger().info('Trying OpenCV without specifying V4L2 backend...')
            for device_id in [0, 1, 2, 3, 4, 5, 6, 7]:
                try:
                    self.get_logger().info(f'Trying device {device_id} with default backend...')
                    cap = cv2.VideoCapture(device_id)
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.get_logger().info(f'Successfully opened device {device_id} with default backend')
                            self.cv2_capture = cap
                            return True
                        else:
                            cap.release()
                            
                except Exception as e:
                    self.get_logger().warn(f'Error trying device {device_id} with default backend: {e}')
                    
            self.get_logger().error('No working video devices found with OpenCV')
            return False
            
        except Exception as e:
            self.get_logger().error(f'OpenCV initialization failed: {e}')
            return False

    def ascii_ify(self, image):
        # Convert numpy array to PIL Image for ascii_magic
        try:
            from PIL import Image as PILImage
            
            # Ensure image is in the right format
            if image is None:
                return "[No image data]"
                
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Handle different image shapes
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # RGB image - keep red channel separate for color mapping
                    red_channel = image[:, :, 0]  # Extract red channel for intensity mapping
                    # Convert to grayscale for ASCII processing using all channels
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    image_rgb = np.stack([gray, gray, gray], axis=2)
                elif image.shape[2] == 4:
                    # RGBA image - extract red channel and convert to grayscale
                    red_channel = image[:, :, 0]
                    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
                    image_rgb = np.stack([gray, gray, gray], axis=2)
                else:
                    return f"[Unsupported channel count: {image.shape[2]}]"
            elif len(image.shape) == 2:
                # Grayscale image - use as both intensity and red channel
                red_channel = image
                image_rgb = np.stack([image, image, image], axis=2)
            else:
                return f"[Unsupported image shape: {image.shape}]"
            
            # Store red channel for later color mapping
            self.red_channel_for_color = red_channel
            
            # Convert to PIL Image
            pil_image = PILImage.fromarray(image_rgb)
            
            # Resize to smaller size for ASCII conversion (performance optimization)
            max_size = 150  # Reduced from 200 for better performance
            if pil_image.width > max_size or pil_image.height > max_size:
                pil_image.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
            
            # Enhance brightness and contrast for better ASCII art
            from PIL import ImageEnhance
            
            # Increase contrast to make features more defined
            contrast_enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = contrast_enhancer.enhance(1.8)  # Increase contrast by 80%
            
            # Increase brightness slightly
            brightness_enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = brightness_enhancer.enhance(1.3)  # Increase brightness by 30%
            
            # Optional: Increase sharpness for more defined edges
            sharpness_enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = sharpness_enhancer.enhance(1.5)  # Increase sharpness by 50%
            
            # Create ASCII art from PIL image (enhanced columns for better detail)
            ascii_image = ascii_magic.from_pillow_image(pil_image)
            ascii_text = ascii_image.to_ascii(columns=100, monochrome=True)  # Increased from 80 for more detail
            
            # Convert ASCII text to rendered image
            return self.ascii_text_to_image(ascii_text)
            
        except Exception as e:
            # Fallback to simple ASCII representation
            return None
    
    def ascii_text_to_image(self, ascii_text):
        """Convert ASCII text to a PIL Image with teal-white gradient based on red channel intensity."""
        try:
            from PIL import Image as PILImage, ImageDraw, ImageFont
            
            # Split ASCII text into lines
            lines = ascii_text.split('\n')
            
            # Calculate image dimensions
            font_size = 8
            try:
                font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', font_size)
            except:
                font = ImageFont.load_default()
            
            # Calculate text dimensions
            max_line_width = max(len(line) for line in lines) if lines else 60
            char_width = font_size * 0.6
            char_height = font_size * 1.2
            
            img_width = int(max_line_width * char_width)
            img_height = int(len(lines) * char_height)
            
            # Create image with black background
            img = PILImage.new('RGB', (img_width, img_height), color='black')
            draw = ImageDraw.Draw(img)
            
            # Get the stored red channel and resize to match ASCII dimensions
            if hasattr(self, 'red_channel_for_color') and self.red_channel_for_color is not None:
                red_resized = cv2.resize(self.red_channel_for_color, (len(lines[0]) if lines else 60, len(lines)), 
                                       interpolation=cv2.INTER_AREA)
            else:
                # Fallback to uniform intensity
                red_resized = np.ones((len(lines), len(lines[0]) if lines else 60)) * 128
            
            # Draw each line of ASCII text with color based on red channel intensity
            y_pos = 0
            for line_idx, line in enumerate(lines):
                x_pos = 0
                for char_idx, char in enumerate(line):
                    if char_idx < red_resized.shape[1] and line_idx < red_resized.shape[0]:
                        # Get red intensity (0-255) and normalize to 0-1
                        red_intensity = red_resized[line_idx, char_idx] / 255.0
                        
                        # Mix teal (0, 128, 128) and white (255, 255, 255) based on red intensity
                        # More red = more teal, less red = more white
                        teal_color = np.array([0, 128, 128])
                        white_color = np.array([255, 255, 255])
                        
                        # Blend colors: higher red intensity = more teal
                        mixed_color = teal_color * red_intensity + white_color * (1 - red_intensity)
                        color = tuple(mixed_color.astype(int))
                    else:
                        # Default to white for characters outside red channel bounds
                        color = (255, 255, 255)
                    
                    # Draw individual character with calculated color
                    draw.text((x_pos, y_pos), char, fill=color, font=font)
                    x_pos += char_width
                y_pos += char_height
            
            # Convert PIL image to numpy array (RGB format)
            img_array = np.array(img)
            return img_array
            
        except Exception as e:
            # Create a simple error image
            error_img = np.zeros((100, 400, 3), dtype=np.uint8)
            return error_img
    
    def publish_camera_feed(self):
        # Check if any camera is initialized
        if self.camera_type is None:
            # Only log warning if cooldown period has passed
            current_time = time.time()
            if current_time - self.last_warning_time >= self.warning_cooldown:
                self.get_logger().warn('No camera initialized, skipping frame capture')
                self.last_warning_time = current_time
            return
            
        image = None
        try:
            # Capture frame based on camera type
            if self.camera_type == 'picamera2':
                image = self.picamera2.capture_array()
                # Even though we configured RGB888, let's verify the channel order
                # Some hardware implementations might return BGR despite RGB888 config
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Test: try both RGB and BGR to see which gives correct colors
                    # For now, assume it's actually BGR and convert to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            elif self.camera_type == 'cv2':
                ret, image = self.cv2_capture.read()
                if not ret:
                    self.get_logger().error('Failed to capture frame from OpenCV camera')
                    # Only reinitialize on actual camera capture failure
                    self.reinitialize_camera()
                    return
                # OpenCV gives us BGR, convert to RGB for ROS message
                if len(image.shape) == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
            else:
                self.get_logger().error(f'Unknown camera type: {self.camera_type}')
                return
            
            # Rotate image 180 degrees (camera is mounted upside down)
            image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_180)
                
        except Exception as e:
            self.get_logger().error(f'Error capturing camera frame: {e}')
            # Only reinitialize on actual camera capture failure
            self.reinitialize_camera()
            return
            
        # If we got here, camera capture was successful, now try to publish
        self.frame_counter += 1
        
        try:
            # Optimized: Skip resizing if camera is already configured to 1920x1080
            height, width = image_rgb.shape[:2]
            
            # Only resize if image is larger than target (should be rare with optimized config)
            if width > 1920 or height > 1080:
                scale = min(1920 / width, 1080 / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create standard ROS Image message (optimized for speed)
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            msg.height = image_rgb.shape[0]
            msg.width = image_rgb.shape[1]
            msg.encoding = 'rgb8'  # Standard ROS encoding
            msg.is_bigendian = False
            msg.step = image_rgb.shape[1] * 3  # 3 bytes per pixel for RGB
            msg.data = image_rgb.flatten().tobytes()  # Flatten to ensure proper byte order
            
            self.publisher_.publish(msg)
            
            # Reduce logging frequency for performance
            if self.frame_counter % 30 == 0:  # Log every 30 frames (once per second at 30fps)
                data_size = len(msg.data)
                self.get_logger().info(f'Published RGB camera feed from {self.camera_type} ({msg.width}x{msg.height}) - '
                                     f'Data size: {data_size/1024:.1f}KB - Frame #{self.frame_counter}')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing camera feed: {e}')
            # Don't reinitialize camera for publishing errors
            return
            
        # ASCII conversion optimization: Only process every Nth frame
        if self.frame_counter % self.ascii_frame_skip == 0:
            try:
                ascii_image_array = self.ascii_ify(image_rgb)
                if ascii_image_array is not None:
                    # Create ROS Image message for ASCII art
                    ascii_msg = Image()
                    ascii_msg.header.stamp = self.get_clock().now().to_msg()
                    ascii_msg.header.frame_id = "ascii_frame"
                    ascii_msg.height = ascii_image_array.shape[0]
                    ascii_msg.width = ascii_image_array.shape[1]
                    ascii_msg.encoding = 'rgb8'
                    ascii_msg.is_bigendian = False
                    ascii_msg.step = ascii_image_array.shape[1] * 3
                    ascii_msg.data = ascii_image_array.flatten().tobytes()
                    
                    self.publisher_ascii_.publish(ascii_msg)
                    self.get_logger().info(f'Published ASCII art image ({ascii_msg.width}x{ascii_msg.height}) - Frame #{self.frame_counter}')
                else:
                    self.get_logger().warn('ASCII conversion returned None')
            except Exception as e:
                self.get_logger().warn(f'ASCII art conversion failed: {e}')
                # Don't reinitialize camera for ASCII conversion errors
            
    def reinitialize_camera(self):
        """Helper method to reinitialize camera after actual camera failures."""
        if self.camera_type == 'picamera2':
            try:
                self.picamera2.close()
            except:
                pass
            self.picamera2 = None
        elif self.camera_type == 'cv2':
            try:
                self.cv2_capture.release()
            except:
                pass
            self.cv2_capture = None
        
        self.camera_type = None
        self.get_logger().info('Attempting to reinitialize camera...')
        self.initialize_camera()
    
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