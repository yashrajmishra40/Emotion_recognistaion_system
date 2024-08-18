import cv2
from deepface import DeepFace

# Load the pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

# Define emotion labels
emotion_labels = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # No faces detected; skip this frame
        cv2.imshow('Real-time Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest) in the original colored frame
        face_roi = frame[y:y + h, x:x + w]

        # Resize the face ROI to match the input shape of the model (48x48)
        resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

        # Convert the resized face image to RGB (3 channels)
        rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)

        # Normalize the RGB face image
        normalized_face = rgb_face / 255.0

        # Convert the image to float32
        float_face = normalized_face.astype('float32')

        # Check the shape of float_face
        print(f"Shape of input image: {float_face.shape}")

        # Reshape the image to match the input shape of the model (1, 48, 48, 3)
        reshaped_face = float_face.reshape(1, 48, 48, 3)

        try:
            # Predict emotions using the pre-trained model
            preds = model.predict(reshaped_face)
            emotion_idx = preds.argmax()
            emotion = emotion_labels[emotion_idx]

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error during prediction: {e}")

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
