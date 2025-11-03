import threading
from functools import partial
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.properties import ObjectProperty
import cv2
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
from PIL import Image as PILImage
from kivy.core.window import Window
import pandas as pd
from time import sleep
from ultralytics import YOLO
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from camera_manager import camera_manager

Window.clearcolor = (.8, .8, .8, 1)

# ----------------- Kivy Screens -----------------
class MainWindow(Screen):
    camera_display = ObjectProperty(None)

class AttendenceWindow(Screen):
    pass

class DatasetWindow(Screen):
    camera_display = ObjectProperty(None)

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("my.kv")

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

    # Camera related
    capture = None
    camera_texture = None
    current_screen = "main"

    # Models
    yolo_model = None
    liveness_model = None
    face_recognizer = None

    # ----------------- Load Models -----------------
    def load_yolo(self):
        if self.yolo_model is None:
            model_path = os.path.join(self.Dir, "yolov8n-face-lindevs.pt")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
            self.yolo_model = YOLO(model_path)
            print("YOLO model loaded successfully")

    def load_liveness(self):
        if self.liveness_model is None:
            live_path = os.path.join(self.Dir, "liveness_model.keras")
            if not os.path.exists(live_path):
                raise FileNotFoundError(f"Liveness model not found at {live_path}")
            self.liveness_model = load_model(live_path)
            print("Liveness model loaded successfully")

    def load_face_recognizer(self):
        if self.face_recognizer is None:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            trainer_path = os.path.join(self.Dir, 'trainer', 'trainer.yml')
            if os.path.exists(trainer_path):
                self.face_recognizer.read(trainer_path)
                print("Face recognizer loaded successfully")
            else:
                print("No trained face recognizer found")

    # ----------------- Helper: YOLO Detection -----------------
    def detect_faces_yolo(self, frame):
        try:
            img = frame[..., ::-1]  # BGR → RGB
            results = self.yolo_model(img, verbose=False)
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    w, h = x2 - x1, y2 - y1
                    # Ensure coordinates are within frame bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    w = x2 - x1
                    h = y2 - y1
                    if w > 0 and h > 0:  # Only add valid boxes
                        boxes.append((x1, y1, w, h))
            return boxes
        except Exception as e:
            print("YOLO detection error:", e)
            return []

    # ----------------- Helper: Liveness -----------------
    def is_live(self, face_crop):
        try:
            if face_crop.size == 0:
                return False, 0.0

            # Convert to grayscale (1 channel)
            if len(face_crop.shape) == 3:
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_crop

            # Resize to match model input size
            gray_face = cv2.resize(gray_face, (96, 96))  # Use 96x96 if that's what your model expects

            # Normalize and prepare for model
            gray_face = gray_face.astype("float") / 255.0
            gray_face = img_to_array(gray_face)
            gray_face = np.expand_dims(gray_face, axis=0)

            # Make prediction
            preds = self.liveness_model.predict(gray_face, verbose=0)
            confidence = float(preds[0][0])
            return confidence > 0.5, confidence
        except Exception as e:
            print("Liveness error:", e)
            return False, 0.0

    # ----------------- Frame Display in Kivy -----------------
    def display_frame(self, frame, screen_name="main"):
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip frame horizontally for mirror effect
            frame_rgb = cv2.flip(frame_rgb, 1)

            # Create texture
            buf = frame_rgb.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

            # Update display in main thread
            Clock.schedule_once(partial(self.update_texture, texture, screen_name))

        except Exception as e:
            print("Display frame error:", e)

    def update_texture(self, texture, screen_name, dt):
        try:
            if screen_name == "main":
                screen = kv.get_screen('main')
                if hasattr(screen, 'ids') and 'camera_display' in screen.ids:
                    screen.ids.camera_display.texture = texture
            elif screen_name == "second":
                screen = kv.get_screen('second')
                if hasattr(screen, 'ids') and 'camera_display' in screen.ids:
                    screen.ids.camera_display.texture = texture
        except Exception as e:
            print("Update texture error:", e)

    # ----------------- UI Helpers -----------------
    def message_cleaner(self):
        while True:
            if not self.msg_clear:
                while self.msg_timer > 0:
                    sleep(0.25)
                    self.msg_timer -= 0.25
                try:
                    kv.get_screen('main').ids.info.text = ""
                    kv.get_screen('second').ids.info.text = ""
                except:
                    pass
                self.msg_clear = True

    def show_message(self, message, screen="both"):
        if (self.msg_thread is None) or not (self.msg_thread.is_alive()):
            self.msg_thread = threading.Thread(target=self.message_cleaner, daemon=True)
            self.msg_thread.start()
        if screen == "both":
            kv.get_screen('main').ids.info.text = message
            kv.get_screen('second').ids.info.text = message
        elif screen == "main":
            kv.get_screen('main').ids.info.text = message
        elif screen == "second":
            kv.get_screen('second').ids.info.text = message
        self.msg_timer = 5
        self.msg_clear = False

    # ----------------- Kivy Build -----------------
    def build(self):
        self.icon = self.Dir + '/webcam.ico'
        self.title = 'Face Detection Attendance System'

        # Preload models
        try:
            self.load_yolo()
            self.load_liveness()
            self.load_face_recognizer()
        except Exception as e:
            print("Model loading warning:", e)

        return kv

    def on_stop(self):
        self.break_loop()

    def break_loop(self):
        self.running = False
        if self.capture and self.capture.isOpened():
            self.capture.release()
        cv2.destroyAllWindows()

    # ----------------- Threads -----------------
    def startAttendence(self):
        if self.att_thread is not None and self.att_thread.is_alive():
            self.show_message("Attendance system already running", "main")
            return
        self.att_thread = threading.Thread(target=self.Attendence, daemon=True)
        self.att_thread.start()
        self.show_message("Starting attendance system...", "main")

    def startTrain(self):
        if self.train_thread is not None and self.train_thread.is_alive():
            self.show_message("Training already in progress", "main")
            return
        self.train_thread = threading.Thread(target=self.train, daemon=True)
        self.train_thread.start()
        self.show_message("Starting training...", "main")

    def startDataset(self):
        if self.data_thread is not None and self.data_thread.is_alive():
            self.show_message("Dataset collection already running", "second")
            return
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
        """Main attendance tracking function with confidence display"""
        try:
            self.load_yolo()
            self.load_liveness()
            self.load_face_recognizer()

            # Load user data
            users_file = os.path.join(self.Dir, 'list', 'users.csv')
            if os.path.exists(users_file):
                users_df = pd.read_csv(users_file)
                id_to_name = dict(zip(users_df['id'], users_df['name']))
            else:
                id_to_name = {}
                self.show_message("No users found. Please create users first.", "main")
                return

            # Test if camera opened successfully
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.show_message("Cannot access camera- trying alternative index","main")
                return
            # Try different camera indices
            self.capture = cv2.VideoCapture(1)
            if not self.capture.isOpened():
                self.show_message("No camera found!", "main")
                return

            # Test camera read
            ret, test_frame = self.capture.read()
            if not ret:
               self.show_message("Camera read failed", "main")
               return

            print(f"Camera initialized: {self.capture.get(3)}x{self.capture.get(4)}")


            # Set camera resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

            self.show_message("Attendance system started. Press 'Stop System' to exit.", "main")

            # Font settings for display
            font = cv2.FONT_HERSHEY_DUPLEX

            while self.running:
                ret, frame = self.capture.read()
                if not ret:
                    self.show_message("Camera error", "main")
                    break

                # Detect faces with YOLO
                boxes = self.detect_faces_yolo(frame)

                for (x, y, w, h) in boxes:
                    # Extract face region
                    face_roi = frame[y:y+h, x:x+w]

                    if face_roi.size == 0:
                        continue

                    # Liveness detection with confidence
                    is_live, live_confidence = self.is_live(face_roi)

                    if not is_live:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, f"FAKE ({live_confidence:.2f})", (x, y-10),
                                   font, 0.7, (0, 0, 255), 2)
                        continue

                    # Face recognition
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    gray_face = cv2.resize(gray_face, (100, 100))

                    id, confidence = self.face_recognizer.predict(gray_face)
                    match_confidence = 100 - confidence  # Convert to percentage

                    if confidence < 70:  # Confidence threshold
                        name = id_to_name.get(id, f"User_{id}")
                        user_id_str = str(id)

                        # Mark attendance if not already marked today
                        if user_id_str not in recognized_today:
                            current_time = datetime.now().strftime("%H:%M:%S")
                            new_entry = pd.DataFrame({
                                'id': [id],
                                'name': [name],
                                'date': [today],
                                'time': [current_time]
                            })
                            attendance_df = pd.concat([attendance_df, new_entry], ignore_index=True)
                            attendance_df.to_csv(attendance_file, index=False)
                            recognized_today.add(user_id_str)

                            self.show_message(f"Attendance marked: {name} ({match_confidence:.1f}%)", "main")
                            print(f"Attendance marked for {name} at {current_time} with {match_confidence:.1f}% confidence")

                        # Green rectangle for recognized face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name}", (x+5, y-35), font, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {match_confidence:.1f}%", (x+5, y-10), font, 0.6, (255, 255, 0), 2)
                        cv2.putText(frame, "LIVE", (x+w-80, y-10), font, 0.6, (0, 255, 0), 2)

                    else:
                        # Blue rectangle for unknown face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, "Unknown", (x+5, y-35), font, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"Confidence: {match_confidence:.1f}%", (x+5, y-10), font, 0.6, (255, 255, 0), 2)
                        cv2.putText(frame, "LIVE", (x+w-80, y-10), font, 0.6, (0, 255, 0), 2)

                # Display frame in Kivy
                self.display_frame(frame, "main")

                # Small delay to prevent high CPU usage
                sleep(0.03)

            if self.capture and self.capture.isOpened():
                self.capture.release()

            self.show_message("Attendance system stopped", "main")

        except Exception as e:
            self.show_message(f"Attendance error: {str(e)}", "main")
            print("Attendance error:", e)
            if self.capture and self.capture.isOpened():
                self.capture.release()

    # ----------------- DATASET COLLECTION -----------------
    def dataset(self):
        """Dataset collection function with confidence display"""
        try:
            self.load_yolo()

            user_id = kv.get_screen('second').ids.user_id.text.strip()
            user_name = kv.get_screen('second').ids.user_name.text.strip()

            if not user_id or not user_name:
                self.show_message("Please enter both ID and Name", "second")
                return

            if not user_id.isdigit():
                self.show_message("User ID must be a number", "second")
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
                self.show_message("User ID already exists", "second")
                return

            # Add new user
            new_user = pd.DataFrame({'id': [user_id], 'name': [user_name]})
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(users_file, index=False)

            # Capture images
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.show_message("Cannot access camera", "second")
                return

            # Set camera resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            dataset_path = os.path.join(self.Dir, 'dataset')
            os.makedirs(dataset_path, exist_ok=True)

            count = 0
            self.running = True

            # Font for display
            font = cv2.FONT_HERSHEY_DUPLEX

            self.show_message("Capturing dataset... Look at camera. Press 'Stop Collection' when done.", "second")

            while self.running and count < 100:  # Capture 100 images max
                ret, frame = self.capture.read()
                if not ret:
                    break

                boxes = self.detect_faces_yolo(frame)

                frame_display = frame.copy()  # Create a copy for display

                if boxes:
                    for (x, y, w, h) in boxes:
                        # Green rectangle for face detection
                        cv2.rectangle(frame_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        face_roi = frame[y:y+h, x:x+w]

                        if face_roi.size == 0:
                            continue

                        # Save face image
                        if count < 100:
                            filename = f"User_{user_id}_{count}.jpg"
                            filepath = os.path.join(dataset_path, filename)
                            cv2.imwrite(filepath, face_roi)
                            count += 1

                        # Display information
                        cv2.putText(frame_display, f"Captured: {count}/100", (x, y-60), font, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame_display, f"User: {user_name}", (x, y-35), font, 0.6, (255, 0, 0), 2)
                        cv2.putText(frame_display, f"ID: {user_id}", (x, y-10), font, 0.6, (255, 0, 0), 2)
                else:
                    cv2.putText(frame_display, "No face detected", (10, 30), font, 0.7, (0, 0, 255), 2)

                # Show count on frame
                cv2.putText(frame_display, f"Images Captured: {count}/100", (10, frame_display.shape[0]-10), font, 0.7, (255, 0, 0), 2)

                self.display_frame(frame_display, "second")
                sleep(0.1)  # Slow down capture rate

            if self.capture and self.capture.isOpened():
                self.capture.release()

            self.show_message(f"Dataset collected: {count} images for {user_name}", "second")
            print(f"Dataset collection complete: {count} images for {user_name}")

        except Exception as e:
            self.show_message(f"Dataset error: {str(e)}", "second")
            print("Dataset error:", e)
            if self.capture and self.capture.isOpened():
                self.capture.release()

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
                    print(f"Failed to read {imagePath}, skipping")
                    continue

                results = self.yolo_model(img[..., ::-1], verbose=False)
                try:
                    id = int(os.path.split(imagePath)[-1].split("_")[1])
                except:
                    print(f"Skipping {imagePath}, cannot parse ID")
                    continue

                detected_faces = 0
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                        if x1 < x2 and y1 < y2:
                            face_crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                            face_crop = cv2.resize(face_crop, (100, 100))
                            faceSamples.append(face_crop)
                            ids.append(id)
                            detected_faces += 1

                if detected_faces == 0:
                    print(f"No faces detected in {imagePath}")

            if len(faceSamples) == 0:
                self.show_message("No faces found for training.", "main")
                return

            recog.train(faceSamples, np.array(ids))
            recog.write(os.path.join(trainer_path, 'trainer.yml'))

            unique_ids = len(np.unique(ids))
            self.show_message(f"Training complete for {unique_ids} faces.", "main")
            print(f"Training complete. {unique_ids} unique IDs, {len(faceSamples)} total samples.")

            # Reload the trained recognizer
            self.load_face_recognizer()

        except Exception as e:
            self.show_message("Error occurred during training.", "main")
            print("Training error:", e)

# ----------------- Run -----------------
if __name__ == "__main__":
    MainApp().run()
