import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os

# Camera
cap = cv2.VideoCapture(0)

# Hand detector
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

label = "a"
folder = f"datas/{label}"
os.makedirs(folder, exist_ok=True)

# Count existing images
existing_files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
counter = len(existing_files)

print(f"Starting from image number: {counter + 1}",f"for label:{label}")

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # FIX: clamp crop so it never goes out of frame bounds
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgresizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)

            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgresizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)

    # Save image
    if key == ord('s') and hands:
        counter += 1
        filename = f"{folder}/{label}_{counter}.jpg"
        cv2.imwrite(filename, imgWhite)
        print(f"Saved {filename}")

    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()