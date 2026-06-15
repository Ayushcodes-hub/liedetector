import tkinter as tk
import cv2
import math
import threading
import time
from PIL import Image, ImageTk
import mediapipe as mp
import speech_recognition as sr

class LieDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("AI Multimodal Lie Detector Engine")
        self.window.geometry("700x920")
        self.window.configure(bg="#121212")

        # 1. TOP SECTION: The Question Area
        self.question_label = tk.Label(
            window, 
            text="QUESTION: Did you eat the last slice of cake?", 
            font=("Arial", 16, "bold"), 
            fg="#FFD700", 
            bg="#1E1E1E", 
            wraplength=600,
            pady=15
        )
        self.question_label.pack(fill=tk.X, padx=20, pady=15)

        # 2. MIDDLE SECTION: The Camera Feed Window
        self.camera_canvas = tk.Canvas(window, width=640, height=480, bg="#2A2A2A", highlightthickness=0)
        self.camera_canvas.pack(pady=10)

        # Analytics overlay panel
        self.analytics_label = tk.Label(
            window,
            text="Blinks: 0 | L EAR: 0.00 | R EAR: 0.00",
            font=("Consolas", 12, "bold"),
            fg="#00FFCC",
            bg="#121212"
        )
        self.analytics_label.pack(pady=2)

        # NEW: FINAL DECISION PANEL
        self.verdict_label = tk.Label(
            window,
            text="VERDICT: ANALYZING BIOMETRIC FEED...",
            font=("Arial", 14, "bold"),
            fg="#FFFFFF",
            bg="#222222",
            width=50,
            pady=8
        )
        self.verdict_label.pack(pady=5)

        # 3. BOTTOM SECTION: The Live Transcript Tab
        self.transcript_title = tk.Label(window, text="LIVE TRANSCRIPT (SPEAK YOUR ANSWER NOW)", font=("Arial", 10, "bold"), fg="#FF3366", bg="#121212")
        self.transcript_title.pack(anchor="w", padx=30, pady=(5,0))
        
        self.transcript_box = tk.Text(
            window, 
            height=3, 
            width=75, 
            font=("Arial", 12), 
            bg="#1E1E1E", 
            fg="#FFFFFF", 
            insertbackground="white",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.transcript_box.pack(padx=20, pady=5)
        
        # --- MULTIMODAL TRACKING VARIABLES ---
        self.blink_count = 0
        self.eye_was_closed = False
        self.EAR_THRESHOLD = 0.22
        
        # Timers to measure behavioral duration stress
        self.start_time = time.time()
        self.last_speech_time = time.time()
        self.user_has_spoken = False
        self.current_transcript = ""

        # --- VOICE PROCESSING INIT ---
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
        self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.audio_callback)

        # Hardware & AI Initialization
        self.video_capture = cv2.VideoCapture(0)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, 
            refine_landmarks=True, 
            min_detection_confidence=0.6, 
            min_tracking_confidence=0.6
        )
        
        self.update_camera_feed()

    def audio_callback(self, recognizer, audio):
        """Fires every time a complete chunk of speech finishes speaking"""
        try:
            transcript = recognizer.recognize_google(audio).lower().strip()
            self.current_transcript = transcript
            self.user_has_spoken = True
            self.last_speech_time = time.time()
            
            # Update transcription UI box instantly
            self.window.after(0, self.update_transcript_ui, transcript.capitalize())
            
            # Run data verification pipeline 
            self.window.after(0, self.run_lie_detection_matrix, transcript)
        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass

    def update_transcript_ui(self, text):
        self.transcript_box.delete("1.0", tk.END)
        self.transcript_box.insert(tk.END, text)

    def calculate_ear(self, landmarks, upper_pts, lower_pts, left_corner, right_corner, img_w, img_h):
        p_left = landmarks[left_corner]
        p_right = landmarks[right_corner]
        horizontal_dist = math.hypot((p_left.x - p_right.x) * img_w, (p_left.y - p_right.y) * img_h)

        vertical_dist_sum = 0
        for up, down in zip(upper_pts, lower_pts):
            p_up = landmarks[up]
            p_down = landmarks[down]
            vertical_dist_sum += math.hypot((p_up.x - p_down.x) * img_w, (p_up.y - p_down.y) * img_h)
        
        avg_vertical_dist = vertical_dist_sum / len(upper_pts)
        if horizontal_dist == 0: return 0.0
        return avg_vertical_dist / horizontal_dist

    def run_lie_detection_matrix(self, statement):
        """Combines speech patterns and eye metrics to calculate a truth probability value"""
        elapsed_time = max(1, time.time() - self.start_time)
        blinks_per_minute = (self.blink_count / elapsed_time) * 60

        # Behavioral Stress Weight
        # Normal relaxed conversational blink rate is roughly 15-25 blinks per minute.
        # Sudden freeze (staring intensely while making a lie up) or rapid blinking signals high stress metrics.
        is_stressed_eyes = blinks_per_minute < 8 or blinks_per_minute > 35

        # Factual Content Extraction Layer
        # Checks if statement contains standard deceptive indicators or admission vectors
        deceptive_words = ["i don't know", "maybe", "i swear", "honestly", "to be honest", "not me", "forget"]
        contains_evasion = any(word in statement for word in deceptive_words)
        
        says_no = "no" in statement or "did not" in statement or "didn't" in statement
        says_yes = "yes" in statement or "i did" in statement or "yeah" in statement

        # Decision Tree Logic
        if says_no:
            if is_stressed_eyes or contains_evasion:
                self.verdict_label.config(text="VERDICT: 🚨 LIE DETECTED (Factual Conflict & Biometric Stress)", fg="#FF3333", bg="#331111")
            else:
                self.verdict_label.config(text="VERDICT: ✅ TRUTH (Biometrics Stable & Consistent)", fg="#33FF33", bg="#113311")
        elif says_yes:
            self.verdict_label.config(text="VERDICT: 📝 FACTUAL CONFISSION ACKNOWLEDGED", fg="#00FFFF", bg="#113333")
        else:
            if is_stressed_eyes:
                self.verdict_label.config(text="VERDICT: ⚠️ EVASIVE ANSWER & HIGH EYE STRESS", fg="#FFCC00", bg="#332200")
            else:
                self.verdict_label.config(text="VERDICT: 🤔 UNCERTAIN STATEMENT (Need clear Yes/No)", fg="#FFFFFF", bg="#222222")

    def update_camera_feed(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img_h, img_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Double eye measurement indices
                left_upper, left_lower = [160, 159, 158], [144, 145, 153]
                right_upper, right_lower = [387, 386, 385], [373, 374, 380]
                
                left_ear = self.calculate_ear(landmarks, left_upper, left_lower, 33, 133, img_w, img_h)
                right_ear = self.calculate_ear(landmarks, right_upper, right_lower, 362, 263, img_w, img_h)
                avg_ear = (left_ear + right_ear) / 2
                
                # Render tracking landmarks over both eyes
                for idx in left_upper + left_lower + [33, 133] + right_upper + right_lower + [362, 263]:
                    cx, cy = int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)
                    cv2.circle(rgb_frame, (cx, cy), 2, (0, 255, 200), -1)

                # Blink Detection State Engine
                if avg_ear < self.EAR_THRESHOLD:
                    if not self.eye_was_closed:
                        self.eye_was_closed = True
                else:
                    if self.eye_was_closed:
                        self.blink_count += 1
                        self.eye_was_closed = False
                
                self.analytics_label.config(
                    text=f"Blinks: {self.blink_count}  |  Left EAR: {left_ear:.2f}  |  Right EAR: {right_ear:.2f}"
                )
            
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.camera_canvas.img_tk = img_tk
            
        self.window.after(15, self.update_camera_feed)

    def __del__(self):
        if hasattr(self, 'stop_listening'):
            self.stop_listening(wait_for_stop=False)
        if hasattr(self, 'video_capture') and self.video_capture.isOpened():
            self.video_capture.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = LieDetectorApp(root)
    root.mainloop()