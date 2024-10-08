import os
import face_recognition
import pickle
from sklearn import neighbors

faces_dir = './data/faces/'
model_save_path = './models/face_recognition_model.pkl'
n_neighbors = 2

face_encodings = []
face_labels = []

if not os.path.exists(faces_dir):
    raise FileNotFoundError(f"The directory {faces_dir} does not exist.")

for student_folder in os.listdir(faces_dir):
    student_path = os.path.join(faces_dir, student_folder)
    if os.path.isdir(student_path):
        for img_file in os.listdir(student_path):
            img_path = os.path.join(student_path, img_file)
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = face_recognition.load_image_file(img_path)
                face_encodings_in_img = face_recognition.face_encodings(img)
                
                if face_encodings_in_img:
                    face_encoding = face_encodings_in_img[0]
                    face_encodings.append(face_encoding)
                    face_labels.append(student_folder)
                else:
                    print(f"No faces found in {img_file}.")

if len(face_encodings) == 0:
    raise ValueError("No face encodings found. Check if the faces directory contains valid images.")

knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree')
knn_clf.fit(face_encodings, face_labels)

model_dir = os.path.dirname(model_save_path)
os.makedirs(model_dir, exist_ok=True)

with open(model_save_path, 'wb') as f:
    pickle.dump(knn_clf, f)

print("Face recognition model trained and saved.")
