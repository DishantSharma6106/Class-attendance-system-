import cv2
import time
from threading import Lock
import logging

class CameraManager:
    """
    Handles all camera operations with proper error handling and resource management
    """

    def __init__(self):
        self.capture = None
        self.camera_lock = Lock()
        self.current_index = 0
        self.is_initialized = False
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Setup logging for camera operations"""
        logger = logging.getLogger('CameraManager')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def find_working_camera(self, max_index=5):
        """
        Find the first working camera index
        Returns: camera_index or -1 if none found
        """
        self.logger.info("🔍 Searching for working camera...")

        for i in range(max_index):
            try:
                self.logger.info(f"Testing camera index {i}")
                cap = cv2.VideoCapture(i)

                if cap.isOpened():
                    # Try to read a frame multiple times
                    for attempt in range(5):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            self.logger.info(f"✅ Camera found at index {i} - Resolution: {frame.shape[1]}x{frame.shape[0]}")
                            cap.release()
                            return i
                        time.sleep(0.1)

                    self.logger.warning(f"Camera {i} opened but no frames received")
                else:
                    self.logger.info(f"❌ Cannot open camera index {i}")

                cap.release()

            except Exception as e:
                self.logger.error(f"Error testing camera {i}: {e}")

        self.logger.error("❌ No working cameras found")
        return -1

    def initialize_camera(self, camera_index=0, width=640, height=480):
        """
        Initialize camera with proper error handling
        Returns: True if successful, False otherwise
        """
        with self.camera_lock:
            try:
                # Release existing camera
                if self.capture and self.capture.isOpened():
                    self.capture.release()

                self.logger.info(f"Initializing camera index {camera_index}")
                self.capture = cv2.VideoCapture(camera_index)

                if not self.capture.isOpened():
                    self.logger.error(f"Failed to open camera index {camera_index}")
                    return False

                # Give camera time to initialize
                time.sleep(1.0)

                # Test frame capture
                ret, frame = self.capture.read()
                if not ret:
                    self.logger.error("Camera opened but cannot read frames")
                    self.capture.release()
                    return False

                self.logger.info(f"✅ Camera initialized: {frame.shape[1]}x{frame.shape[0]}")

                # Set desired resolution
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                self.current_index = camera_index
                self.is_initialized = True
                return True

            except Exception as e:
                self.logger.error(f"Camera initialization error: {e}")
                return False

    def get_frame(self):
        """
        Get a frame from the camera
        Returns: (success, frame) tuple
        """
        with self.camera_lock:
            if not self.is_initialized or not self.capture or not self.capture.isOpened():
                self.logger.warning("Camera not initialized")
                return False, None

            try:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    self.logger.warning("Failed to capture frame")
                    return False, None

                return True, frame

            except Exception as e:
                self.logger.error(f"Frame capture error: {e}")
                return False, None

    def auto_initialize(self, preferred_index=0, fallback_search=True):
        """
        Automatically initialize camera with fallback
        Returns: True if successful, False otherwise
        """
        self.logger.info("🚀 Auto-initializing camera...")

        # Try preferred index first
        if self.initialize_camera(preferred_index):
            return True

        # If preferred fails and fallback is enabled, search for any working camera
        if fallback_search:
            self.logger.info("Preferred camera failed, searching for alternatives...")
            working_index = self.find_working_camera()
            if working_index != -1:
                return self.initialize_camera(working_index)

        return False

    def get_camera_info(self):
        """Get camera information"""
        if not self.is_initialized or not self.capture:
            return "Camera not initialized"

        try:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            return f"Index: {self.current_index}, Resolution: {width}x{height}, FPS: {fps:.1f}"
        except:
            return "Camera info unavailable"

    def release_camera(self):
        """Safely release camera resources"""
        with self.camera_lock:
            if self.capture and self.capture.isOpened():
                self.capture.release()
                self.logger.info("✅ Camera released")
            self.is_initialized = False

    def is_camera_available(self):
        """Check if camera is available and working"""
        if not self.is_initialized or not self.capture:
            return False

        # Quick test
        ret, frame = self.get_frame()
        return ret


# Singleton instance for easy access
camera_manager = CameraManager()


# Test function
def test_camera_manager():
    """Test the camera manager"""
    print("Testing Camera Manager...")

    # Auto-initialize camera
    if camera_manager.auto_initialize():
        print("✅ Camera initialized successfully")
        print(f"📊 Camera info: {camera_manager.get_camera_info()}")

        # Test frame capture
        for i in range(5):
            success, frame = camera_manager.get_frame()
            if success:
                print(f"✅ Frame {i+1}: {frame.shape}")
            else:
                print(f"❌ Failed to get frame {i+1}")
            time.sleep(0.1)

        camera_manager.release_camera()
    else:
        print("❌ Failed to initialize camera")


if __name__ == "__main__":
    test_camera_manager()
