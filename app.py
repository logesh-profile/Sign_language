import cv2
import numpy as np
import math
import os
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Paths
TFLITE_PATH = "Model/model.tflite"
LABELS_PATH = "Model/labels.txt"

# Load TFLite model
interpreter = None
if os.path.exists(TFLITE_PATH):
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded!")

# Load mediapipe hand detector
import mediapipe as mp
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

offset = 20
imgSize = 300
labels = ['0','1','2','3','4','5','6','7','8','9',
          'a','b','c','d','e','f','g','h','i','j',
          'k','l','m','n','o','p','q','r','s','t',
          'u','v','w','x','y','z','delete','space']

CONFIDENCE_THRESHOLD = 0.80

def detect_hand_bbox(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if results.multi_hand_landmarks:
        h, w = img.shape[:2]
        lm = results.multi_hand_landmarks[0].landmark
        xs = [l.x * w for l in lm]
        ys = [l.y * h for l in lm]
        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))
        return x1, y1, x2 - x1, y2 - y1
    return None

@app.route('/')
def index():
    return render_template('live.html')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"sign": ""}), 400

        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None or interpreter is None:
            return jsonify({"sign": ""})

        bbox = detect_hand_bbox(img)
        if bbox:
            x, y, w, h = bbox
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                aspectRatio = h / w if w != 0 else 0
                try:
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    imgInput = cv2.resize(imgWhite, (224, 224))
                    imgInput = imgInput.astype(np.float32) / 255.0
                    imgInput = np.expand_dims(imgInput, axis=0)

                    interpreter.set_tensor(input_details[0]['index'], imgInput)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

                    index = int(np.argmax(prediction))
                    confidence = float(prediction[index])

                    if confidence > CONFIDENCE_THRESHOLD:
                        return jsonify({"sign": labels[index]})
                except Exception as e:
                    print(f"Detection error: {e}")

        return jsonify({"sign": ""})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"sign": ""}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
