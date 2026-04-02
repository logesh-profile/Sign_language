import cv2
import numpy as np
import math
import os
import base64
from flask import Flask, render_template, request, jsonify

# Suppress tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)

# Model files
has_model = os.path.exists("Model/keras_model.h5") and os.path.exists("Model/labels.txt")
detector = HandDetector(maxHands=1) if has_model else None
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt") if has_model else None

offset = 20
imgSize = 300
labels = ['0','1','2','3','4','5','6','7','8','9',
          'a','b','c','d','e','f','g','h','i','j',
          'k','l','m','n','o','p','q','r','s','t',
          'u','v','w','x','y','z','delete','space']

CONFIDENCE_THRESHOLD = 0.80

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

        # Extract base64 image data (remove the 'data:image/jpeg;base64,' prefix)
        img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        
        # Decode base64 to OpenCV image
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"sign": ""})

        if detector and classifier:
            hands, _ = detector.findHands(img, draw=False)
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                
                # Boundary checks for cropping
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
                            imgInput = cv2.resize(imgWhite, (224, 224))
                            prediction, index = classifier.getPrediction(imgInput, draw=False)
                        else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize
                            imgInput = cv2.resize(imgWhite, (224, 224))
                            prediction, index = classifier.getPrediction(imgInput, draw=False)
                            
                        if prediction is not None:
                            confidence = prediction[index]
                            if confidence > CONFIDENCE_THRESHOLD:
                                return jsonify({"sign": labels[index]})
                    except Exception as e:
                        pass # Silently handle any resize/prediction exceptions

        return jsonify({"sign": ""})
    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({"sign": ""}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)