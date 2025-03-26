import cv2
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Load the trained model
model = tf.keras.models.load_model("violence_detection_model.h5")

# Input image size used during training
IMG_SIZE = 128

# Load video (or use webcam with cap = cv2.VideoCapture(0))
cap = cv2.VideoCapture("NV_1.mp4")

# Detection threshold and frame count limit
THRESHOLD = 0.5
VIOLENCE_TRIGGER_COUNT = 5
consecutive_violence_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess frame
    resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    normalized = resized_frame.astype('float32') / 255.0
    input_data = np.expand_dims(normalized, axis=0)

    # Predict violence score (sigmoid output)
    prediction = model.predict(input_data, verbose=0)
    violence_score = prediction[0][0]

    # Count consecutive violent frames
    if violence_score > THRESHOLD:
        consecutive_violence_count += 1
    else:
        consecutive_violence_count = 0  # reset if a safe frame is found

    # Alert if enough violent frames
    if consecutive_violence_count >= VIOLENCE_TRIGGER_COUNT:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        alpha = 0.3
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.putText(frame, "!!! VIOLENCE DETECTED !!!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
    else:
        cv2.putText(frame, "No Violence", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Violence Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
