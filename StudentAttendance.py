import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)  # opens web cam

video_capture.set(3, 640)
video_capture.set(4, 480)

# Directory containing the face images
faces_dir = "Face_Reco_imgs/"

imgBackground = cv2.imread('Resources/background-img.png')

# Initialize lists to store encodings and names
known_face_encodings = []
known_faces_names = []

# Load all the images from the directory
for filename in os.listdir(faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        image_encoding = face_recognition.face_encodings(image)[0]

        # Add the encoding and the name (without extension) to the lists
        known_face_encodings.append(image_encoding)
        known_faces_names.append(os.path.splitext(filename)[0])

print(known_faces_names)

students = known_faces_names.copy()

# --------------------------------------------

face_locations = []  # to save the face coming from webcam
face_encodings = []  # to capture characteristics of persons faces: shape, facial landmark, texture
face_names = []  # to store names of faces present in the list --> (known_faces_names)

# to get exact date, month, year
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# creating csv file
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    success, frame = video_capture.read()  # reads info from the webcam
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # resize image to 1/4th of original image
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # color conversion from bgr to rgb

    face_locations = face_recognition.face_locations(rgb_small_frame)  # detect faces from webcam
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # encode the detected faces
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)  # stricter matching
        name = "Unknown"  # Default to "Unknown" if no match is found

        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)  # calculate face distances
        best_match_index = np.argmin(face_distance)  # find the best match

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]  # match found, use the corresponding name

        face_names.append(name)  # add the name to the list

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M-%S")
            lnwriter.writerow([name, current_time])
            print(f"{name} recognized at {current_time}")

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

        # Draw a filled rectangle below the face to display the name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0) if name != "Unknown" else (0, 0, 255), cv2.FILLED)

        # Display the name and add a bold font style
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 5, bottom - 6), font, 0.8, (255, 255, 255), 2)

    # Update imgBackground with the modified frame
    imgBackground[162:162 + 480, 55:55 + 640] = frame

    # Display the resulting image
    cv2.imshow("Attendance System", imgBackground)

    # Quit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()