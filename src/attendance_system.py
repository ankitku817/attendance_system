import tkinter as tk
from tkinter import messagebox
import cv2
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime, time
import face_recognition

with open('models/face_recognition_model.pkl', 'rb') as f:
    face_recognition_model = pickle.load(f)

emotion_model = load_model('models/emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
attendance = []
logged_students = set()

def log_attendance(name, emotion):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attendance.append({"Name": name, "Emotion": emotion, "Time": current_time})
    logged_students.add(name)

def save_to_csv():
    if attendance:
        df = pd.DataFrame(attendance)
        df.to_csv('attendance.csv', index=False)
        print("Attendance saved to 'attendance.csv'")
    else:
        print("No attendance data to save.")

def is_in_time_window():
    now = datetime.now().time()
    start_time = time(9, 30)
    end_time = time(10, 0)
    return start_time <= now <= end_time

def recognize_face(face_encoding):
    try:
        name = face_recognition_model.predict([face_encoding])[0]
        return name
    except Exception as e:
        print(f"Error recognizing face: {e}")
        return "Unknown"

def predict_emotion(face_image):
    face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image_resized = cv2.resize(face_image_gray, (48, 48))
    face_image_resized = np.expand_dims(face_image_resized, axis=-1)
    face_image_rgb = np.repeat(face_image_resized, 3, axis=-1)
    face_image_rgb = np.expand_dims(face_image_rgb, axis=0)
    face_image_rgb = face_image_rgb / 255.0
    prediction = emotion_model.predict(face_image_rgb)
    return emotion_labels[np.argmax(prediction)]

def start_attendance_system():
    cap = cv2.VideoCapture(0)
    while True:
        current_time = datetime.now().time()
        if not is_in_time_window():
            messagebox.showinfo("Information", "Attendance time window is closed.")
            break
        
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Unable to capture video.")
            break
        
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            top, right, bottom, left = face_location
            face_img = frame[top:bottom, left:right]
            name = recognize_face(face_encoding)
            if name != "Unknown" and name not in logged_students:
                emotion = predict_emotion(face_img)
                log_attendance(name, emotion)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f'{name} - {emotion}', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_to_csv()

root = tk.Tk()
root.title("Attendance System")
start_button = tk.Button(root, text="Start Attendance System", command=start_attendance_system, font=("Arial", 14))
start_button.pack(pady=20)
root.mainloop()
