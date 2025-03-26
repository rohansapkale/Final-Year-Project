from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("violence_detection_model.h5")

# Model input size
IMG_SIZE = 128

# Prediction threshold
THRESHOLD = 0.6

# For counting consecutive violence predictions
consecutive_violence_count = 0
VIOLENCE_TRIGGER_COUNT = 5  # Only trigger alert if 5+ consecutive frames predict violence
current_status = "Normal"

# Initialize webcam
cap = cv2.VideoCapture(0)

def gen_frames():
    global current_status, consecutive_violence_count

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Preprocess the frame
        resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype('float32') / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # Model prediction
        prediction = model.predict(input_data, verbose=0)[0][0]
        print(f"Prediction Score: {prediction:.4f}")

        # Track violence prediction
        if prediction >= THRESHOLD:
            consecutive_violence_count += 1
        else:
            consecutive_violence_count = 0

        # Determine status
        if consecutive_violence_count >= VIOLENCE_TRIGGER_COUNT:
            status_text = "⚠️ ALERT: Violence Detected!"
            current_status = "violence"
            color = (0, 0, 255)
        else:
            status_text = "Normal"
            current_status = "nonviolence"
            color = (0, 255, 0)

        # Display on video
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"status": current_status})

if __name__ == '__main__':
    app.run(debug=True)
