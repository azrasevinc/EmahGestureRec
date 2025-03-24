import finroc
import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pyttsx3 
import speech_recognition as sr
# from gtts import gTTS #Für die Sprachausgabe
from textblob import TextBlob
from pydub import AudioSegment
import time

class RobotLearns(finroc.FinrocModule):
    def __init__(self, name):
        super().__init__(name)
        self.image_port = self.create_input_port("tImageVector", "image")
        self.detected_gesture = self.create_output_port("String", "detected_gesture")
        self.robot_response = self.create_output_port("String", "robot_response")
        self.human_sentiment = self.create_output_port("String", "human_sentiment")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.csv_path = '/mnt/public/learning/trained_models/torch/image_based/mimic_gesture/coordinatesUni.csv'
        self.model_path = '/mnt/public/learning/trained_models/torch/image_based/mimic_gesture/modellNeu.pkl'
        self.feedback_csv_path= '/mnt/public/learning/trained_models/torch/image_based/mimic_gesture/feedback.csv'
        #self.train_model() #zuerst modell trainieren
        self.load_model(self.model_path)  # Lade das trainierte Modell und den Scaler
        self.df=pd.read_csv(self.csv_path) 
        self.speak = ""
        self.init_feedback_csv()

    def init_feedback_csv (self):
        #initialisiere die CSV-Datei
        if not os.path.exists (self.feedback_csv_path):
            with open(self.feedback_csv_path, 'w', newline = '') as csvfile: 
                writer= csv.writer (csvfile)
                writer.writerow (["timestamp", "detected_gesture", "feedback", "Sentiment"])
                print("Feedback CSV-Datei initalisieren.")
    
    def save_feedback( self, detected_gesture, feedback, sentiment): 
        #Speichert das Feedback in eine CSV Datei
        try:
            with open (self.feedback_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([pd.Timestamp.now(), detected_gesture, feedback, sentiment])
                print ("Feedback gespeichert")
        except Exception as e:
            print(f"Fehler beim Speichern des Feedbacks:{e}")
    
    def update_csv(self, gesture, feedback, sentiment):
        try:
            
            if gesture:
                 self.save_feedback (gesture, feedback, sentiment)
        except Exception as e:
            print (f"Fehler update Text{e}")
      

    def train_model(self):
        print ("Training des Models gestartet.")
        """Trainiert ein Machine-Learning-Modell mit den gesammelten Daten."""
        try:
            df = pd.read_csv(self.csv_path)
            X = df.drop('class', axis=1)
            y = df['class']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Modell-Pipeline erstellen
            self.model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
            self.model.fit(X_train, y_train)

            # Modell bewerten
            y_pred = self.model.predict(X_test)
            genauigkeit = accuracy_score(y_test, y_pred)
            print(f"Modellgenauigkeit: {genauigkeit:.2f}")

            # Modell speichern
            self.save_object(self.model, self.model_path)

        except Exception as e:
            print(f"Fehler beim Trainieren des Modells: {e}")

    def load_model(self,filename) :
        """Lädt das trainierte Modell."""
        try:
            with open (filename, 'rb') as f:
                self.model = pickle.load(f)
                print("Modell geladen.")
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {e}")

    def save_object(self, obj, filename):
        #Speichert ein Objekt in einer Datei.
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
    
            
    def get_audio_feedback(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=15)
            try:
                # audio = recognizer.listen(source, timeout=10) 
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            except sr.UnknownValueError:
                print("Audio input not understood.")
                return None
            except sr.RequestError as e:
                print(f"Error in speech recognition: {e}")
                return None

    def analyze_feedback(self, feedback):
        #Analysiert das Feedback und gibt 'positive', 'negative' oder 'neutral' zurück."""
        if not feedback:
            return "neutral"
        blob = TextBlob(feedback)
        sentiment = blob.sentiment.polarity
       
        # Here, add logic to convert polarity into 'positive', 'negative', or 'neutral'
        if sentiment > 0:
            self.human_sentiment.Publish("positive", 0)
            return "positive"
            
        elif sentiment < 0:
            self.human_sentiment.Publish("negative", 0)
            return "negative"
        
        else:
            self.human_sentiment.Publish("neutral", 0)
            return "neutral"
            

    def process_frame(self, frame):
        #Verarbeitet einen Frame und sagt die Geste vorher.
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
                # Landmark-Daten extrahieren
                landmark_data = {}
                for part in ["pose", "face", "left_hand", "right_hand"]:
                    landmarks = getattr(results, f"{part}_landmarks")
                    landmark_data[part] = []
                    num_landmarks = 33 if part == "pose" else (468 if part == "face" else 21)
                    for lmk in range(num_landmarks):
                        if landmarks:
                            landmark = landmarks.landmark[lmk]
                            landmark_data[part].extend([landmark.x, landmark.y, landmark.z])
                            if part == "pose":  # Nur Visibility für Pose-Landmarks hinzufügen
                                landmark_data[part].append(landmark.visibility)
                        else:
                            landmark_data[part].extend([0.0, 0.0, 0.0])  # Platzhalter für x, y, z
                            if part == "pose":
                                landmark_data[part].append(0.0)  # Platzhalter für Visibility

                # Daten zu einer Liste zusammenfügen
                row = landmark_data["pose"] + landmark_data["face"] + landmark_data["left_hand"] + landmark_data["right_hand"]
                print(f"Länge von 'row': {len(row)}")

                #Spaltennamen aus der CSV-Datei lesen
                
                # Daten skalieren
                X = pd.DataFrame([row], columns= self.df.columns[1:])

                # Gestik vorhersagen
                gestik_klasse = self.model.predict(X)[0]
                gestik_wahrscheinlichkeiten = self.model.predict_proba(X)[0]
                #self.detected_gesture.Publish(gestik_klasse, 0)

                # Ergebnis anzeigen
                cv2.putText(image, f"Gestik: {gestik_klasse}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Rückgabewerte für die run-Methode
                #return gestik_klasse, gestik_wahrscheinlichkeiten 

            except Exception as e:
                print(f"Fehler während der Vorhersage: {e}")

        cv2.imshow('zed feed', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
            
        ## Rückgabewerte für die run-Methode
        return gestik_klasse, gestik_wahrscheinlichkeiten 
        # return True

    def run(self):
        try: 
            finroc.Initialize("gestureMitFeedback")
            print("Press 's' and Enter to start processing a gesture, press 'q' and Enter to quit.")
            gesture_count = 0
            should_listen = False
            versuche = 0  # Make sure this is managed correctly across cycles

            while gesture_count < 15:  # Limit to 11 gestures
                key_input = input("Waiting for 's' to start or 'q' to quit: ")
                if key_input == 'q':
                    break
                elif key_input == 's':
                    self.robot_response.Publish("", 0) 
                    self.detected_gesture.Publish("", 0)
                    self.human_sentiment.Publish("", 0)
                    if should_listen:
                        feedback = self.get_audio_feedback()
                        sentiment = self.analyze_feedback(feedback)
                        self.update_csv(gestik_klasse, feedback, sentiment)
                        print(f"Feedback: {feedback}, Sentiment: {sentiment}, Attempt: {versuche}")

                        if sentiment == "positive":
                            speak = "Super! Teach me the next."
                            self.robot_response.Publish(speak, 0)
                            should_listen = False  # Ensure no listening next
                            gesture_count += 1
                            versuche = 0  # Reset attempts when moving to next gesture
                        elif sentiment == "neutral":
                            speak = "Could you please give me some constructive feedback?"
                            self.robot_response.Publish(speak, 0)
                            versuche += 1
                            if versuche >= 3:
                                speak = "Moving to the next gesture."
                                self.robot_response.Publish(speak, 0)
                                should_listen = False
                                gesture_count += 1
                                versuche = 0  # Reset attempts
                        elif sentiment == "negative":
                            if versuche < 2:
                                alternative_geste = self.df['class'][top_indices[versuche]]
                                self.detected_gesture.Publish(alternative_geste, 0)
                                speak = f"Perhaps, did you mean '{alternative_geste}'?"
                                self.robot_response.Publish(speak, 0)
                                versuche += 1
                            else:
                                speak = "Let's try a new gesture."
                                self.robot_response.Publish(speak, 0)
                                should_listen = False
                                gesture_count += 1
                                versuche = 0
                    else:
                        self.update()
                        if self.image_port.HasChanged():
                            images = self.image_port.Get()
                            if len(images) > 0:
                                img = images[0]
                                self.image_port.ResetChanged()
                                frame = np.copy(img[..., :3])
                                gestik_klasse, gestik_wahrscheinlichkeiten = self.process_frame(frame)
                                self.detected_gesture.Publish(gestik_klasse, 0)
                                if gestik_klasse:
                                    top_indices = np.argsort(gestik_wahrscheinlichkeiten)[-3:][::-1]
                                    speak = f"Is this the gesture '{gestik_klasse}'?"
                                    self.robot_response.Publish(speak, 0)
                                    should_listen = True
                                    versuche = 0
                                print("End of processing for one gesture.")
                        else:
                            print("No new images to process.")
                else:
                    print("Invalid input, please press 's' to process a gesture or 'q' to exit.")
            self.robot_response.Publish("goodbye", 0) 
        except Exception as e:
            print(f"An error occurred: {e}")




if __name__ == "__main__":
    module = RobotLearns("gestureMitFeedback")
    module.run()