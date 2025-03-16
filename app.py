from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import base64
import numpy as np
from deepface import DeepFace
from collections import deque
import time
import logging
import os

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define stress-related emotions
stress_emotions = ['angry', 'sad', 'fear', 'surprise']
emotion_buffer = deque(maxlen=10)

# Variables for timing warnings
multi_face_start_time = None
no_face_start_time = None
WARNING_THRESHOLD = 5

# Route for the UI
@app.route('/')
def index():
    return render_template('index.html')

# Route for emotion analysis
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    global multi_face_start_time, no_face_start_time

    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            logger.error("No image data received")
            return jsonify({"error": "No image data received"}), 400

        try:
            # Decode the base64 image
            img_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return jsonify({"error": "Invalid image data"}), 400

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        # Check for no face or multiple faces
        warning_message = None
        if len(faces) == 0:
            if no_face_start_time is None:
                no_face_start_time = time.time()

            elapsed_time = time.time() - no_face_start_time
            if elapsed_time >= WARNING_THRESHOLD:
                warning_message = "No face detected for 5 seconds!"
        else:
            no_face_start_time = None

        if len(faces) > 1:
            if multi_face_start_time is None:
                multi_face_start_time = time.time()

            elapsed_time = time.time() - multi_face_start_time
            if elapsed_time >= WARNING_THRESHOLD:
                warning_message = "Multiple faces detected for 5 seconds!"
        else:
            multi_face_start_time = None

        if warning_message:
            logger.warning(warning_message)
            return jsonify({"warning": warning_message, "status": "error"})

        # Analyze emotions for each detected face
        results = []
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            try:
                # Use DeepFace to analyze emotions
                predictions = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                detected_emotion = predictions[0]['dominant_emotion']
                confidence = predictions[0]['emotion'][detected_emotion]

                # Add detected emotion to buffer
                if confidence > 60:
                    emotion_buffer.append(detected_emotion)

                # Determine the most frequent emotion in the buffer
                stable_emotion = max(set(emotion_buffer), key=emotion_buffer.count) if emotion_buffer else "neutral"
                stress_status = "STRESSED" if stable_emotion in stress_emotions else "Calm"

                # Append results
                results.append({
                    "emotion": stable_emotion,
                    "stress": stress_status,
                    "bounding_box": {
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h)
                    }
                })

                logger.info(f"Result: {results}")

            except Exception as e:
                logger.error(f"DeepFace Error: {e}")

        return jsonify({"results": results})

    except Exception as e:
        logger.error(f"Server Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True)