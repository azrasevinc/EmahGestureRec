import finroc
import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd
import pickle

class RobotLearns(finroc.FinrocModule):
    def __init__(self, name):
        super().__init__(name)
        self.image_port = self.create_input_port("tImageVector", "image")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.csv_path = 'Path to test.csv' #change for each sentence coordinates_1.csv
        self.Samples_pro_Klasse= 27 #increase based on sentence length
        self.collected_samples = 0 


    def collect_landmarks(self, class_name, frame):
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:             
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        
                results = holistic.process(image)
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw face landmarks with specific color and style
                self.mp_drawing.draw_landmarks(
                    image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                    self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),  # Greenish lines
                    self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)  # Light green points
                )
                
                # Draw right hand landmarks with specific color and style
                self.mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),  # Darker reddish lines
                    self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)  # Violet points
                )
                
                # Draw left hand landmarks with specific color and style
                self.mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),  # Dark pinkish lines
                    self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)  # Lighter pink points
                )
                
                # Draw pose landmarks with specific color and style
                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),  # Orange lines
                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # Purple points
                )
                try:
                    # Landmark-Daten in einem Dictionary speichern
                    landmark_data = {}
                    for part in ["pose", "face", "left_hand", "right_hand"]:
                        landmarks = getattr(results, f"{part}_landmarks")
                        landmark_data[part] = []
                        num_landmarks = 33 if part == "pose" else (468 if part == "face" else 21)
                        for lmk in range(num_landmarks):
                            if landmarks:
                                landmark = landmarks.landmark[lmk]
                                landmark_data[part].extend([landmark.x, landmark.y, landmark.z])
                                if part == "pose":  
                                    landmark_data[part].append(landmark.visibility)
                            else:
                                landmark_data[part].extend([0.0, 0.0, 0.0])  # Platzhalter für x, y, z
                                if part == "pose":
                                    landmark_data[part].append(0.0)  # Platzhalter für Visibility

                    # Daten in CSV-Datei schreiben (Reihenfolge beachten!)
                    with open(self.csv_path, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        row = landmark_data["pose"] + landmark_data["face"] + landmark_data["left_hand"] + landmark_data["right_hand"]
                        csv_writer.writerow([class_name] + row)
                    print(f"Gesammelte Samples für Klasse '{class_name}': {self.collected_samples}", end='\r')

                except Exception as e:
                    print(f"Fehler: {e}")

                # Example of setting the window size to fit within the screen resolution
                screen_width = 1280  # Screen width in pixels
                screen_height = 800  # Screen height in pixels

                # Resize the image to fit within the screen size (keeping the aspect ratio)
                height, width = image.shape[:2]

                # Compute scaling factor to fit the image into the screen resolution
                scaling_factor = min(screen_width / width, screen_height / height)

                # Resize the image based on the scaling factor
                resized_image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))

                # Display resized image in a window
                cv2.imshow('zed feed', resized_image)

                # Resize window to fit the resized image size
                cv2.resizeWindow('zed feed', int(width * scaling_factor), int(height * scaling_factor))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    return False
            
        return True

    def run(self):
        finroc.Initialize("gestureMitFeedback")
        while self.collected_samples < self.Samples_pro_Klasse:
            self.update()
            if self.image_port.HasChanged():
                images = self.image_port.Get()
                if len(images) > 0:
                    img = images[0]
                    self.image_port.ResetChanged()
                    frame = np.copy(img[..., :3])
                    if not self.collect_landmarks("test", frame): #change for each sentence p01, p02 test
                        break
                    self.collected_samples +=1
                    #self.process_frame(frame, scaler, model)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    module = RobotLearns("gestureMitFeedback")
    module.run()
 