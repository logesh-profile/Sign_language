import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import threading
import time
import tempfile
from collections import deque
from gtts import gTTS
import pygame

# suppress tensorflow per-step logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────
#  TTS ENGINE — gTTS + pygame
# ─────────────────────────────────────────────
pygame.mixer.init()
_tts_lock = threading.Lock()

def speak_text(text):
    def _speak():
        with _tts_lock:
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmp_path = tmp.name
                tmp.close()
                tts = gTTS(text=text, lang='en', tld='com.au', slow=False)
                tts.save(tmp_path)
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                pygame.mixer.music.unload()
                os.remove(tmp_path)
            except Exception as e:
                print(f"TTS error: {e}")
    threading.Thread(target=_speak, daemon=True).start()

# ─────────────────────────────────────────────
#  CAMERA & MODELS
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
detector   = HandDetector(maxHands=1)
Classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset  = 20
imgSize = 300

labels = ['0','1','2','3','4','5','6','7','8','9',
          'a','b','c','d','e','f','g','h','i','j',
          'k','l','m','n','o','p','q','r','s','t',
          'u','v','w','x','y','z','delete','space']

# ─────────────────────────────────────────────
#  THRESHOLDS
# ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80
HOLD_FRAMES          = 12

# ─────────────────────────────────────────────
#  LIGHT CONDITION PREPROCESSING
#
#  3 modes — auto detected every frame:
#
#  NIGHT  (brightness < 80)
#    → strong CLAHE clipLimit=4.0
#    → gamma boost (brightens dark hand)
#    → white balance correction (removes yellow LED cast)
#
#  NORMAL (80 <= brightness <= 175)
#    → imgCrop returned completely unchanged
#    → zero processing, zero risk to existing accuracy
#
#  GLARE  (brightness > 175)
#    → mild CLAHE clipLimit=2.0 on L channel only
#    → reduces bright hotspots
# ─────────────────────────────────────────────
NIGHT_THRESHOLD = 80
GLARE_THRESHOLD = 175

clahe_normal = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
clahe_night  = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))

# Gamma LUT — precomputed once for speed
def _make_gamma_lut(gamma):
    return np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).astype(np.uint8)

gamma_lut = _make_gamma_lut(2.0)   # brightens dark images

def white_balance(img):
    """Simple gray-world white balance — removes colour cast from LED light."""
    result = img.copy().astype(np.float32)
    avg_b  = np.mean(result[:, :, 0])
    avg_g  = np.mean(result[:, :, 1])
    avg_r  = np.mean(result[:, :, 2])
    avg    = (avg_b + avg_g + avg_r) / 3.0
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg / avg_r), 0, 255)
    return result.astype(np.uint8)

def adjust_light(img_crop):
    """
    Auto-detect light condition and apply the right correction.
    Normal light → completely unchanged (protects existing accuracy).
    """
    gray       = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    # ── NIGHT / dim LED ──────────────────────
    if brightness < NIGHT_THRESHOLD:
        # Step 1: white balance — remove yellow LED cast
        corrected = white_balance(img_crop)
        # Step 2: CLAHE on L channel — enhance local contrast
        lab      = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        l_eq     = clahe_night.apply(l)
        lab_eq   = cv2.merge((l_eq, a, b))
        corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        # Step 3: gamma boost — brighten the hand
        corrected = cv2.LUT(corrected, gamma_lut)
        return corrected, "night"

    # ── GLARE / bright sunlight ───────────────
    elif brightness > GLARE_THRESHOLD:
        lab      = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)
        l, a, b  = cv2.split(lab)
        l_eq     = clahe_normal.apply(l)
        lab_eq   = cv2.merge((l_eq, a, b))
        corrected = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return corrected, "glare"

    # ── NORMAL — completely untouched ────────
    else:
        return img_crop, "normal"

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
sentence          = ""
stable_counter    = 0
last_prediction   = ""
letter_flash      = 0
prediction_buffer = deque(maxlen=5)
light_mode        = "normal"

# ─────────────────────────────────────────────
#  DESIGN SYSTEM — "Obsidian Glass"
# ─────────────────────────────────────────────
C_BG      = ( 18,  20,  26)
C_PANEL   = ( 24,  27,  35)
C_CARD    = ( 34,  38,  50)
C_CARD2   = ( 44,  48,  62)
C_DIV     = ( 55,  60,  78)
C_WHITE   = (228, 230, 236)
C_MUTED   = (105, 110, 130)
C_ACCENT  = (200, 150,  55)
C_TEAL    = (160, 210, 130)
C_RED     = ( 80,  90, 210)
C_FLASH   = (180, 220, 255)
C_ORANGE  = ( 50, 165, 255)
C_NIGHT   = (180, 120,  40)   # deep blue badge for night mode

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTD = cv2.FONT_HERSHEY_DUPLEX

# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────
def fill_rrect(img, pt1, pt2, color, r=10):
    x1,y1=pt1; x2,y2=pt2
    r = min(r,(x2-x1)//2,(y2-y1)//2)
    cv2.rectangle(img,(x1+r,y1),(x2-r,y2),color,-1)
    cv2.rectangle(img,(x1,y1+r),(x2,y2-r),color,-1)
    for cx,cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img,(cx,cy),r,color,-1)

def stroke_rrect(img, pt1, pt2, color, t=1, r=10):
    x1,y1=pt1; x2,y2=pt2
    r = min(r,(x2-x1)//2,(y2-y1)//2)
    cv2.line(img,(x1+r,y1),(x2-r,y1),color,t)
    cv2.line(img,(x1+r,y2),(x2-r,y2),color,t)
    cv2.line(img,(x1,y1+r),(x1,y2-r),color,t)
    cv2.line(img,(x2,y1+r),(x2,y2-r),color,t)
    for (cx,cy,a1,a2) in [(x1+r,y1+r,180,270),(x2-r,y1+r,270,360),
                           (x1+r,y2-r,90,180),(x2-r,y2-r,0,90)]:
        cv2.ellipse(img,(cx,cy),(r,r),0,a1,a2,color,t)

def pbar(img, x, y, w, h, pct, fg, bg=None):
    bg = bg or C_DIV
    fill_rrect(img,(x,y),(x+w,y+h),bg,r=h//2)
    if pct > 0.015:
        fw = max(h, int(w*min(pct,1.0)))
        fill_rrect(img,(x,y),(x+fw,y+h),fg,r=h//2)

def lbl(img, text, x, y, color=None, sc=0.37):
    cv2.putText(img,text.upper(),(x,y),FONT,sc,color or C_MUTED,1,cv2.LINE_AA)

def val(img, text, x, y, color=None, sc=0.72, th=1):
    cv2.putText(img,text,(x,y),FONTD,sc,color or C_WHITE,th,cv2.LINE_AA)

# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (640, 480))
    H, W = 480, 640
    PANEL_W = 390
    TOTAL_W = W + PANEL_W

    canvas = np.zeros((H, TOTAL_W, 3), np.uint8)
    canvas[:] = C_BG

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    prediction  = None
    index       = -1
    confidence  = 0.0
    imgWhite    = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    light_mode  = "normal"

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        # ── Auto light adjustment ────────────
        imgCrop, light_mode = adjust_light(imgCrop)
        # ─────────────────────────────────────

        aspectRatio = h / w if w != 0 else 0

        if aspectRatio > 1:
            k    = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            imgInput = cv2.resize(imgWhite, (224, 224))
            prediction, index = Classifier.getPrediction(imgInput, draw=False)
        else:
            k    = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            imgInput = cv2.resize(imgWhite, (224, 224))
            prediction, index = Classifier.getPrediction(imgInput, draw=False)

        if prediction is not None:
            confidence      = prediction[index]
            predicted_label = labels[index]

            prediction_buffer.append(predicted_label)
            most_common    = max(set(prediction_buffer),
                                 key=prediction_buffer.count)
            majority_count = prediction_buffer.count(most_common)

            if most_common == last_prediction:
                stable_counter += 1
            else:
                stable_counter  = 0
                last_prediction = most_common

            if (confidence >= CONFIDENCE_THRESHOLD and
                    stable_counter == HOLD_FRAMES and
                    majority_count >= 4):

                if most_common == 'delete':
                    sentence = sentence[:-1]
                elif most_common == 'space':
                    sentence += ' '
                else:
                    sentence += most_common
                letter_flash   = 28
                stable_counter = 0
                prediction_buffer.clear()

        else:
            prediction_buffer.clear()

        ok      = confidence >= CONFIDENCE_THRESHOLD
        box_col = C_TEAL if ok else C_RED
        ov      = imgOutput.copy()
        cv2.rectangle(ov,(x-offset-4,y-offset-4),(x+w+offset+4,y+h+offset+4),box_col,5)
        cv2.addWeighted(ov,0.3,imgOutput,0.7,0,imgOutput)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),box_col,2)

        if prediction is not None:
            tag = f"{labels[index].upper()}  {confidence*100:.0f}%"
            (tw,_),_ = cv2.getTextSize(tag,FONTD,0.6,1)
            px2,py2  = x-offset, y-offset-12
            fill_rrect(imgOutput,(px2-4,py2-16),(px2+tw+8,py2+4),
                       C_TEAL if ok else C_RED, r=5)
            cv2.putText(imgOutput,tag,(px2+2,py2-1),FONTD,0.6,(12,14,18),1,cv2.LINE_AA)

    else:
        prediction_buffer.clear()

    canvas[:H, :W] = imgOutput

    # ── PANEL ────────────────────────────────
    PX = W + 16
    PW = PANEL_W - 30

    fill_rrect(canvas,(W,0),(TOTAL_W,H),C_PANEL,r=0)
    cv2.line(canvas,(W,0),(W,H),C_DIV,1)
    fill_rrect(canvas,(W,0),(TOTAL_W,4),C_ACCENT,r=0)

    cv2.putText(canvas,"SignSpeak",(PX,46),FONTD,1.0,C_WHITE,2,cv2.LINE_AA)
    lbl(canvas,"Sign Language  ·  Voice Output",PX,64)
    cv2.line(canvas,(PX,72),(PX+PW,72),C_DIV,1)

    # ── Detected letter card ─────────────────
    cy = 82
    fill_rrect(canvas,(PX,cy),(PX+PW,cy+108),C_CARD,r=12)
    stroke_rrect(canvas,(PX,cy),(PX+PW,cy+108),C_DIV,1,r=12)
    lbl(canvas,"Detected",PX+12,cy+18)

    if prediction is not None and hands:
        big  = labels[index].upper()
        fcol = C_FLASH if letter_flash > 0 else C_WHITE
        (tw,_),_ = cv2.getTextSize(big,FONTD,2.8,3)
        tx = PX + (PW - tw)//2
        cv2.putText(canvas,big,(tx,cy+90),FONTD,2.8,fcol,3,cv2.LINE_AA)
        cv2.circle(canvas,(PX+PW//2,cy+101),3,C_ACCENT,-1)
    else:
        cv2.putText(canvas,"—",(PX+PW//2-12,cy+85),FONTD,2.2,C_DIV,2,cv2.LINE_AA)

    # ── Light mode badge ─────────────────────
    if light_mode == "glare":
        fill_rrect(canvas,(PX+PW-80,cy+2),(PX+PW-2,cy+22),C_ORANGE,r=5)
        cv2.putText(canvas,"GLARE",(PX+PW-74,cy+16),
                    FONT,0.38,(15,15,15),1,cv2.LINE_AA)
    elif light_mode == "night":
        fill_rrect(canvas,(PX+PW-80,cy+2),(PX+PW-2,cy+22),C_NIGHT,r=5)
        cv2.putText(canvas,"NIGHT",(PX+PW-74,cy+16),
                    FONT,0.38,(255,255,255),1,cv2.LINE_AA)

    # ── Confidence bar ───────────────────────
    ry = cy + 122
    lbl(canvas,"Confidence",PX+2,ry)
    cpct = confidence if hands and prediction is not None else 0.0
    ccol = C_TEAL if confidence >= CONFIDENCE_THRESHOLD else C_RED
    pbar(canvas,PX,ry+5,PW,9,cpct,ccol)
    cv2.putText(canvas,f"{confidence*100:.1f}%",(PX+PW-50,ry+2),FONT,0.4,ccol,1,cv2.LINE_AA)

    # ── Stability bar ────────────────────────
    ry2 = ry + 28
    lbl(canvas,"Stability  (hold sign to fill)",PX+2,ry2)
    spct = min(stable_counter/HOLD_FRAMES,1.0)
    scol = C_FLASH if spct > 0.85 else C_ACCENT
    pbar(canvas,PX,ry2+5,PW,9,spct,scol)
    cv2.putText(canvas,f"{int(spct*100)}%",(PX+PW-38,ry2+2),FONT,0.4,scol,1,cv2.LINE_AA)

    cv2.line(canvas,(PX,ry2+22),(PX+PW,ry2+22),C_DIV,1)

    # ── Sentence card ────────────────────────
    sy   = ry2 + 32
    sh   = H - sy - 84
    fbord= C_TEAL if letter_flash > 0 else C_DIV

    fill_rrect(canvas,(PX,sy),(PX+PW,sy+sh),C_CARD,r=12)
    stroke_rrect(canvas,(PX,sy),(PX+PW,sy+sh),fbord,1,r=12)
    lbl(canvas,"Sentence",PX+12,sy+17)

    max_ch  = 18
    disp    = sentence if sentence else ""
    lines   = ([disp[i:i+max_ch] for i in range(0,len(disp),max_ch)]
               if disp else [""])
    max_vis = max(1,(sh-28)//28)
    visible = lines[-max_vis:]
    for li,line in enumerate(visible):
        ty = sy + 36 + li*28
        if ty + 5 < sy + sh:
            cv2.putText(canvas,line,(PX+12,ty),FONTD,0.65,C_WHITE,1,cv2.LINE_AA)

    # Blinking cursor
    if int(time.time()*2)%2==0:
        ll = visible[-1] if visible else ""
        (cw,_),_ = cv2.getTextSize(ll,FONTD,0.65,1)
        cur_x = PX+14+cw
        cur_y = sy+36+(len(visible)-1)*28
        if cur_y+4 < sy+sh:
            fill_rrect(canvas,(cur_x,cur_y-16),(cur_x+2,cur_y+3),C_ACCENT,r=1)

    if letter_flash > 0:
        letter_flash -= 1

    # ── Buttons ──────────────────────────────
    by0 = H - 76
    cv2.line(canvas,(PX,by0),(PX+PW,by0),C_DIV,1)
    bw  = (PW - 8) // 3
    btns= [("S","Speak",C_TEAL),("C","Clear",C_ACCENT),("Q","Quit",C_MUTED)]
    for i,(k2,bl,bc) in enumerate(btns):
        bx  = PX + i*(bw+4)
        bby = by0 + 10
        fill_rrect(canvas,(bx,bby),(bx+bw,bby+36),C_CARD2,r=8)
        stroke_rrect(canvas,(bx,bby),(bx+bw,bby+36),bc,1,r=8)
        fill_rrect(canvas,(bx+7,bby+8),(bx+22,bby+28),bc,r=4)
        cv2.putText(canvas,k2,(bx+10,bby+23),FONTD,0.42,(12,14,18),1,cv2.LINE_AA)
        cv2.putText(canvas,bl,(bx+26,bby+23),FONT,0.42,C_WHITE,1,cv2.LINE_AA)

    cv2.imshow("SignSpeak", canvas)
    cv2.imshow("Hand ROI", imgWhite)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('s'),ord('S')):
        if sentence.strip():
            speak_text(sentence.strip())
    elif key in (ord('c'),ord('C')):
        sentence = ""
    elif key in (ord('q'),ord('Q'),27):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()