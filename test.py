import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model(R'Ypur path to the model')

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for(x,y,w,h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        resized_frame = cv2.resize(face_roi, (48, 48)) 
        img_pixels = resized_frame
        img_pixels = np.expand_dims(img_pixels, axis=0) 

        
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions)
        emotions = ['Happy', 'Sad', 'Neutral']
        predicted_emotion = emotions[max_index]

       
        cv2.putText(frame, predicted_emotion, (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        max_edge = max(w, h)
        x_center = x + w // 2
        y_center = y + h // 2

        
        x_new = x_center - max_edge // 2
        y_new = y_center - max_edge // 2

       
        cv2.rectangle(frame, (x_new, y_new), (x_new + max_edge, y_new + max_edge), (0, 0, 255), 2)
    
    cv2.imshow('Video', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
