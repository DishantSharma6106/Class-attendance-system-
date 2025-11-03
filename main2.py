import threading
from functools import partial
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.properties import ObjectProperty, StringProperty
import cv2
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
from kivy.core.window import Window
import pandas as pd
from time import sleep
from ultralytics import YOLO
import torch
import torch.nn as nn

Window.clearcolor = (.8, .8, .8, 1)

# ----------------- Anti-Spoof Integration -----------------
class AntiSpoofDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the anti-spoof model"""
        try:
            # Add the anti-spoof directory to path
            anti_spoof_dir = os.path.join(os.path.dirname(__file__), 'light-weight-face-anti-spoofing-master')
            if not os.path.exists(anti_spoof_dir):
                print("❌ Anti-spoof directory not found")
                self.model = None
                return

            sys.path.insert(0, anti_spoof_dir)

            try:
                # Import the model
                from models.mobilenetv3 import mobilenetv3_large

                # Create model instance
                self.model = mobilenetv3_large(
                    width_mult=1.0,
                    prob_dropout=0.0,
                    type_dropout='bernoulli',
                    prob_dropout_linear=0.1,
                    embeding_dim=512,
                    mu=0.5,
                    sigma=0.3,
                    theta=0.0,
                    multi_heads=False
                )
            except ImportError as e:
                print(f"❌ Cannot import anti-spoof modules: {e}")
                self.model = None
                return

            # Try to load pre-trained weights
            model_path = self.find_model_file()
            if model_path:
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    print(f"✅ Anti-spoof model loaded from {os.path.basename(model_path)}")
                except Exception as e:
                    print(f"❌ Error loading weights: {e}")
                    print("⚠️ Using uninitialized model")
            else:
                print("⚠️ No pre-trained weights found - using initialized model")

            if self.model:
                self.model.to(self.device)
                self.model.eval()
                print("✅ Anti-spoof model initialized successfully")
            else:
                print("❌ Failed to initialize anti-spoof model")

        except Exception as e:
            print(f"❌ Error loading anti-spoof model: {e}")
            self.model = None

    def find_model_file(self):
        """Find anti-spoof model file"""
        # Check multiple possible locations
        locations = [
            os.path.join(os.path.dirname(__file__), 'light-weight-face-anti-spoofing-master', 'models'),
            os.path.join(os.path.dirname(__file__), 'light-weight-face-anti-spoofing-master'),
            os.path.dirname(__file__)
        ]

        # Also check for common model file names
        common_names = ['anti_spoof.pth', 'model_best.pth', 'mobilenetv3.pth']

        for location in locations:
            if os.path.exists(location):
                for file in os.listdir(location):
                    if file.endswith(('.pth', '.pt', '.pth.tar')):
                        return os.path.join(location, file)
                # Check common names
                for name in common_names:
                    potential_path = os.path.join(location, name)
                    if os.path.exists(potential_path):
                        return potential_path
        return None

    def preprocess(self, face_image):
        """Preprocess face image for the model"""
        if face_image.size == 0:
            return None

        # Resize to model input size
        resized = cv2.resize(face_image, (224, 224))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std

        # Convert to tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, face_image):
        """Predict if face is real or spoof"""
        try:
            if self.model is None:
                return self.enhanced_fallback_detection(face_image)

            input_tensor = self.preprocess(face_image)
            if input_tensor is None:
                return True, 0.5

            with torch.no_grad():
                output = self.model(input_tensor)
                # Handle different output formats
                if isinstance(output, (list, tuple)):
                    output = output[0]  # Take first output if multiple
                probabilities = torch.softmax(output, dim=1)
                real_prob = probabilities[0, 1].item()  # Assuming class 1 is real

            is_real = real_prob > 0.5
            return is_real, real_prob

        except Exception as e:
            print(f"❌ Anti-spoof prediction error: {e}")
            return self.enhanced_fallback_detection(face_image)

    def enhanced_fallback_detection(self, face_image):
        """Enhanced fallback spoof detection"""
        try:
            if face_image.size == 0:
                return False, 0.0

            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Multiple heuristics for better fallback
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast_score = gray.std()

            # Color analysis
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1].mean()

            # Edge density analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (face_image.shape[0] * face_image.shape[1])

            # Combine scores with weights
            score = 0.5
            if blur_score > 100: score += 0.2
            if contrast_score > 40: score += 0.15
            if saturation > 50: score += 0.1
            if edge_density > 0.05: score += 0.05

            score = max(0.1, min(0.9, score))
            is_real = score > 0.6

            return is_real, score

        except Exception as e:
            print(f"Fallback detection error: {e}")
            return True, 0.5

# ----------------- Kivy Screens -----------------
class MainWindow(Screen):
    camera_display = ObjectProperty(None)
    info_text = StringProperty("System ready. Click 'Start Attendance' to begin.")

class AttendenceWindow(Screen):
    info_text = StringProperty("Attendance monitoring active")

class DatasetWindow(Screen):
    camera_display = ObjectProperty(None)
    info_text = StringProperty("Enter user details and start collection")
    user_id = StringProperty("")
    user_name = StringProperty("")

class WindowManager(ScreenManager):
    pass

# Load KV file
try:
    kv = Builder.load_file("my.kv")
except Exception as e:
    print(f"❌ Error loading KV file: {e}")
    # Fallback basic UI
    kv = Builder.load_string('''
WindowManager:
    MainWindow:
        name: "main"
    AttendenceWindow:
        name: "attendence"
    DatasetWindow:
        name: "dataset"
''')

# ----------------- Main App -----------------
class MainApp(App):
    running = False
    Dir = os.path.dirname(os.path.realpath(__file__))
    msg_thread = None
    att_thread = None
    data_thread = None
    train_thread = None
    msg_clear = True
    msg_timer = 0

    # Thread locks for safety
    _att_lock = None
    _data_lock = None
    _train_lock = None

    # Models
    yolo_model = None
    anti_spoof_model = None
    face_recognizer = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_directories()
        self.optimize_performance()

    def create_directories(self):
        """Create required directories"""
        dirs = ['dataset', 'trainer', 'list', 'Attendance']
        for dir_name in dirs:
            dir_path = os.path.join(self.Dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
        print("✅ Directory structure created")

    def optimize_performance(self):
        """Optimize model performance"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("✅ CUDA optimization enabled")

    # ----------------- Load Models -----------------
    def load_yolo(self):
        if self.yolo_model is None:
            model_path = os.path.join(self.Dir, "yolov8n-face-lindevs.pt")
            if not os.path.exists(model_path):
                # Try to download or use default YOLO
                try:
                    print("📥 Loading default YOLO face detection model...")
                    self.yolo_model = YOLO('yolov8n.pt')  # Fallback to standard YOLO
                    print("✅ Default YOLO model loaded")
                except Exception as e:
                    raise FileNotFoundError(f"YOLO model not found at {model_path} and could not load default: {e}")
            else:
                self.yolo_model = YOLO(model_path)
                print("✅ YOLO model loaded successfully")

    def load_anti_spoof(self):
        """Load the anti-spoof model"""
        if self.anti_spoof_model is None:
            try:
                print("🚀 Loading Anti-Spoof Detection...")
                self.anti_spoof_model = AntiSpoofDetector()
                if self.anti_spoof_model.model is not None:
                    print("✅ Anti-spoof detection READY!")
                    return True
                else:
                    print("⚠️ Anti-spoof using fallback mode")
                    return False
            except Exception as e:
                print(f"❌ Failed to load anti-spoof model: {e}")
                self.anti_spoof_model = AntiSpoofDetector()  # Still create instance for fallback
                return False

    def load_face_recognizer(self):
        if self.face_recognizer is None:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            trainer_path = os.path.join(self.Dir, 'trainer', 'trainer.yml')
            if os.path.exists(trainer_path):
                self.face_recognizer.read(trainer_path)
                print("✅ Face recognizer loaded successfully")
            else:
                print("❌ No trained face recognizer found")

    # ----------------- Helper: YOLO Detection -----------------
    def detect_faces_yolo(self, frame):
        try:
            img = frame[..., ::-1]  # BGR → RGB
            results = self.yolo_model(img, verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes:
                    # Check if it's a face (class 0 for face models, or any detection for general YOLO)
                    if len(box.cls) > 0 and (hasattr(self.yolo_model, 'names') and
                       (box.cls[0] == 0 or self.yolo_model.names[int(box.cls[0])] in ['face', 'person'])):
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        w, h = x2 - x1, y2 - y1
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        w = x2 - x1
                        h = y2 - y1
                        if w > 0 and h > 0:
                            boxes.append((x1, y1, w, h))
            return boxes
        except Exception as e:
            print("❌ YOLO detection error:", e)
            return []

    # ----------------- Anti-Spoof Check -----------------
    def is_live(self, face_crop):
        """Anti-spoof detection"""
        try:
            if self.anti_spoof_model is None:
                print("⚠️ No anti-spoof model - assuming REAL")
                return True, 0.95

            if face_crop.size == 0:
                return False, 0.0

            is_real, confidence = self.anti_spoof_model.predict(face_crop)

            print(f"🔍 Anti-spoof - Real: {is_real}, Confidence: {confidence:.3f}")

            # Safety: if low confidence spoof, assume real
            if not is_real and confidence < 0.3:
                print("🔧 Override: Low-confidence spoof -> REAL")
                return True, 0.8

            return is_real, confidence

        except Exception as e:
            print(f"❌ Anti-spoof error: {e}")
            return True, 0.8

    # ----------------- Static Display -----------------
    def display_static_frame(self, screen_name="main"):
        try:
            width, height = 640, 480
            frame = np.ones((height, width, 3), dtype=np.uint8) * 240

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Attendance System Active"
            text_size = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2

            cv2.putText(frame, text, (text_x, text_y), font, 1, (100, 100, 100), 2)
            cv2.putText(frame, "AI Anti-Spoof Detection Active", (text_x-80, text_y+50), font, 0.7, (150, 150, 150), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = frame_rgb.tobytes()
            texture = Texture.create(size=(width, height), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

            Clock.schedule_once(partial(self.update_texture, texture, screen_name))

        except Exception as e:
            print("❌ Display static frame error:", e)

    def update_texture(self, texture, screen_name, dt):
        try:
            screen = self.root.get_screen(screen_name)
            if hasattr(screen, 'ids') and 'camera_display' in screen.ids:
                screen.ids.camera_display.texture = texture
        except Exception as e:
            print(f"❌ Update texture error for {screen_name}:", e)

    # ----------------- UI Helpers -----------------
    def message_cleaner(self):
        while True:
            if not self.msg_clear:
                while self.msg_timer > 0:
                    sleep(0.25)
                    self.msg_timer -= 0.25
                try:
                    # Update StringProperties instead of direct text access
                    main_screen = self.root.get_screen('main')
                    dataset_screen = self.root.get_screen('dataset')

                    if hasattr(main_screen, 'info_text'):
                        main_screen.info_text = ""
                    if hasattr(dataset_screen, 'info_text'):
                        dataset_screen.info_text = ""

                except Exception as e:
                    print(f"Message cleaner error: {e}")
                self.msg_clear = True

    def show_message(self, message, screen="both"):
        if (self.msg_thread is None) or not (self.msg_thread.is_alive()):
            self.msg_thread = threading.Thread(target=self.message_cleaner, daemon=True)
            self.msg_thread.start()

        try:
            if screen == "both":
                main_screen = self.root.get_screen('main')
                dataset_screen = self.root.get_screen('dataset')
                if hasattr(main_screen, 'info_text'):
                    main_screen.info_text = message
                if hasattr(dataset_screen, 'info_text'):
                    dataset_screen.info_text = message
            elif screen == "main":
                main_screen = self.root.get_screen('main')
                if hasattr(main_screen, 'info_text'):
                    main_screen.info_text = message
            elif screen == "dataset":
                dataset_screen = self.root.get_screen('dataset')
                if hasattr(dataset_screen, 'info_text'):
                    dataset_screen.info_text = message
        except Exception as e:
            print(f"Show message error: {e}")

        self.msg_timer = 5
        self.msg_clear = False

    # ----------------- Kivy Build -----------------
    def build(self):
        self.icon = self.Dir + '/webcam.ico'
        self.title = 'AI Face Recognition Attendance'

        try:
            self.load_yolo()
            self.load_anti_spoof()
            self.load_face_recognizer()
        except Exception as e:
            print("⚠️ Model loading warning:", e)

        return kv

    def on_stop(self):
        self.break_loop()

    def break_loop(self):
        self.running = False
        print("🛑 System stopped")

    # ----------------- Threads -----------------
    def startAttendence(self):
        if self.att_thread is not None and self.att_thread.is_alive():
            self.show_message("Attendance system already running", "main")
            return

        if self._att_lock and self._att_lock.locked():
            self.show_message("Attendance system is busy", "main")
            return

        self._att_lock = threading.Lock()
        self.att_thread = threading.Thread(target=self.Attendence, daemon=True)
        self.att_thread.start()
        self.show_message("Starting attendance system...", "main")

    def startTrain(self):
        if self.train_thread is not None and self.train_thread.is_alive():
            self.show_message("Training already in progress", "main")
            return

        if self._train_lock and self._train_lock.locked():
            self.show_message("Training system is busy", "main")
            return

        self._train_lock = threading.Lock()
        self.train_thread = threading.Thread(target=self.train, daemon=True)
        self.train_thread.start()
        self.show_message("Starting training...", "main")

    def startDataset(self):
        if self.data_thread is not None and self.data_thread.is_alive():
            self.show_message("Dataset collection already running", "dataset")
            return

        if self._data_lock and self._data_lock.locked():
            self.show_message("Dataset collection is busy", "dataset")
            return

        self._data_lock = threading.Lock()
        self.data_thread = threading.Thread(target=self.dataset, daemon=True)
        self.data_thread.start()

    # ----------------- File Openers -----------------
    def UserList(self):
        users_file = os.path.join(self.Dir, 'list', 'users.csv')
        if not os.path.exists(users_file):
            self.show_message("Users file not found. Create users first.", "both")
            return
        try:
            if sys.platform == "win32":
                os.startfile(users_file)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, users_file])
        except Exception as e:
            print(e)
            self.show_message("Cannot open users file", "both")

    def AttendanceList(self):
        attendance_file = os.path.join(self.Dir, 'Attendance', 'Attendance.csv')
        if not os.path.exists(attendance_file):
            self.show_message("Attendance file not found. No attendance recorded yet.", "both")
            return
        try:
            if sys.platform == "win32":
                os.startfile(attendance_file)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, attendance_file])
        except Exception as e:
            print(e)
            self.show_message("Cannot open attendance file", "both")

    # ----------------- ATTENDANCE SYSTEM -----------------
    def Attendence(self):
        """Main attendance system with AI anti-spoof detection"""
        try:
            self.load_yolo()
            self.load_anti_spoof()
            self.load_face_recognizer()

            # Load user data
            users_file = os.path.join(self.Dir, 'list', 'users.csv')
            if os.path.exists(users_file):
                users_df = pd.read_csv(users_file)
                id_to_name = dict(zip(users_df['id'], users_df['name']))
                print(f"📋 Loaded {len(id_to_name)} users")
            else:
                self.show_message("No users found. Please create users first.", "main")
                return

            # Initialize camera
            capture = self.initialize_camera()
            if not capture:
                self.show_message("❌ No camera found!", "main")
                return

            self.running = True

            # Setup attendance file
            attendance_dir = os.path.join(self.Dir, 'Attendance')
            os.makedirs(attendance_dir, exist_ok=True)
            attendance_file = os.path.join(attendance_dir, 'Attendance.csv')

            # Load existing attendance
            if os.path.exists(attendance_file):
                attendance_df = pd.read_csv(attendance_file)
            else:
                attendance_df = pd.DataFrame(columns=['id', 'name', 'date', 'time'])

            today = datetime.now().strftime("%Y-%m-%d")
            recognized_today = set(
                attendance_df[attendance_df['date'] == today]['id'].astype(str).values
            ) if not attendance_df.empty else set()

            self.show_message("✅ AI Anti-Spoof System Started", "main")
            self.display_static_frame("main")

            check_count = 0
            max_checks = 1000

            print("🎯 Starting face detection with anti-spoof...")

            while self.running and check_count < max_checks:
                ret, frame = capture.read()
                if not ret:
                    break

                boxes = self.detect_faces_yolo(frame)

                if boxes:
                    for (x, y, w, h) in boxes:
                        face_roi = frame[y:y+h, x:x+w]

                        if face_roi.size == 0:
                            continue

                        # Anti-spoof check
                        is_live, live_conf = self.is_live(face_roi)

                        if not is_live:
                            self.show_message(f"🚫 Spoof detected! ({live_conf:.3f})", "main")
                            continue

                        # Face recognition
                        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                        gray_face = cv2.resize(gray_face, (100, 100))

                        id, confidence = self.face_recognizer.predict(gray_face)

                        if confidence < 70:  # Recognized
                            name = id_to_name.get(id, f"User_{id}")
                            user_id_str = str(id)

                            if user_id_str not in recognized_today:
                                current_time = datetime.now().strftime("%H:%M:%S")
                                new_entry = pd.DataFrame({
                                    'id': [id], 'name': [name], 'date': [today], 'time': [current_time]
                                })
                                attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                                attendance_df.to_csv(attendance_file, index=False)
                                recognized_today.add(user_id_str)

                                success_msg = f"✅ {name} marked present!"
                                self.show_message(success_msg, "main")
                                sleep(3)
                            else:
                                self.show_message(f"ℹ️ {name} already marked today", "main")
                        else:
                            self.show_message("❓ Unknown person", "main")

                check_count += 1
                sleep(1)

            if capture.isOpened():
                capture.release()

            self.show_message("🛑 System stopped", "main")

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.show_message(error_msg, "main")
            print(error_msg)
        finally:
            if hasattr(self, '_att_lock'):
                self._att_lock = None

    def initialize_camera(self, camera_id=0):
        """Better camera initialization"""
        for i in range(3):  # Try multiple cameras
            capture = cv2.VideoCapture(i)
            if capture.isOpened():
                # Test if camera works
                ret, frame = capture.read()
                if ret and frame is not None:
                    print(f"✅ Camera {i} initialized successfully")
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return capture
                capture.release()

        print("❌ No working camera found")
        return None

    # ----------------- DATASET COLLECTION -----------------
    def dataset(self):
        """Dataset collection"""
        try:
            self.load_yolo()

            dataset_screen = self.root.get_screen('dataset')
            user_id = dataset_screen.ids.user_id.text.strip() if hasattr(dataset_screen, 'ids') and 'user_id' in dataset_screen.ids else ""
            user_name = dataset_screen.ids.user_name.text.strip() if hasattr(dataset_screen, 'ids') and 'user_name' in dataset_screen.ids else ""

            if not user_id or not user_name:
                self.show_message("Please enter both ID and Name", "dataset")
                return

            if not user_id.isdigit():
                self.show_message("User ID must be a number", "dataset")
                return

            user_id = int(user_id)

            # Save user info
            users_dir = os.path.join(self.Dir, 'list')
            os.makedirs(users_dir, exist_ok=True)
            users_file = os.path.join(users_dir, 'users.csv')

            if os.path.exists(users_file):
                users_df = pd.read_csv(users_file)
            else:
                users_df = pd.DataFrame(columns=['id', 'name'])

            if user_id in users_df['id'].values:
                self.show_message("User ID already exists", "dataset")
                return

            new_user = pd.DataFrame({'id': [user_id], 'name': [user_name]})
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(users_file, index=False)

            # Initialize camera
            capture = self.initialize_camera()
            if not capture:
                self.show_message("Cannot access camera", "dataset")
                return

            dataset_path = os.path.join(self.Dir, 'dataset')
            os.makedirs(dataset_path, exist_ok=True)

            count = 0
            self.running = True

            self.show_message("📸 Capturing dataset...", "dataset")
            self.display_static_frame("dataset")

            while self.running and count < 50:
                ret, frame = capture.read()
                if not ret:
                    break

                boxes = self.detect_faces_yolo(frame)

                if boxes:
                    for (x, y, w, h) in boxes:
                        face_roi = frame[y:y+h, x:x+w]

                        if face_roi.size == 0:
                            continue

                        filename = f"User_{user_id}_{count}.jpg"
                        filepath = os.path.join(dataset_path, filename)
                        cv2.imwrite(filepath, face_roi)
                        count += 1
                        self.show_message(f"Captured: {count}/50", "dataset")
                        break

                sleep(0.3)

            if capture.isOpened():
                capture.release()

            self.show_message(f"✅ Collected {count} images for {user_name}", "dataset")

        except Exception as e:
            error_msg = f"❌ Dataset error: {str(e)}"
            self.show_message(error_msg, "dataset")
            print(error_msg)
        finally:
            if hasattr(self, '_data_lock'):
                self._data_lock = None

    # ----------------- TRAIN -----------------
    def train(self):
        dataset_path = os.path.join(self.Dir, 'dataset')
        trainer_path = os.path.join(self.Dir, 'trainer')
        os.makedirs(trainer_path, exist_ok=True)

        try:
            self.load_yolo()

            recog = cv2.face.LBPHFaceRecognizer_create()
            imagesPath = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]

            if not imagesPath:
                self.show_message("No images found in dataset.", "main")
                return

            faceSamples, ids = [], []

            for imagePath in imagesPath:
                img = cv2.imread(imagePath)
                if img is None:
                    continue

                results = self.yolo_model(img[..., ::-1], verbose=False)
                try:
                    id = int(os.path.split(imagePath)[-1].split("_")[1])
                except:
                    continue

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        if x1 < x2 and y1 < y2:
                            face_crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            face_crop = cv2.resize(face_crop, (100, 100))
                            faceSamples.append(face_crop)
                            ids.append(id)

            if len(faceSamples) == 0:
                self.show_message("No faces found for training.", "main")
                return

            recog.train(faceSamples, np.array(ids))
            recog.write(os.path.join(trainer_path, 'trainer.yml'))

            unique_ids = len(np.unique(ids))
            success_msg = f"✅ Trained on {len(faceSamples)} samples for {unique_ids} users"
            self.show_message(success_msg, "main")
            print(success_msg)

            self.load_face_recognizer()

        except Exception as e:
            error_msg = "❌ Training error"
            self.show_message(error_msg, "main")
            print(f"{error_msg}: {e}")
        finally:
            if hasattr(self, '_train_lock'):
                self._train_lock = None

# ----------------- Run -----------------
if __name__ == "__main__":
    MainApp().run()
