import cv2
import mediapipe as mp
import numpy as np
import pickle
import cvzone

MODEL_PATH = "model.pk1"

with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
THRESH = float(bundle.get("unknown_threshold", 0.80))

cap = cv2.VideoCapture(0)
detector = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def hand_vec(result):
    if result.multi_hand_landmarks:
        hlm = result.multi_hand_landmarks[0]
        coords = []
        for lm in hlm.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return coords
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (720, 480))
    real_frame = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cvzone.putTextRect(frame, "ASL ABC", (10, 80))

    vec = hand_vec(result)

    label = "unknown"
    conf = 0.0

    if vec is not None:
        X = np.array(vec, dtype=np.float32).reshape(1, -1)
        probs = pipeline.predict_proba(X)[0]
        best_i = int(np.argmax(probs))
        best_prob = float(probs[best_i])
        best_class = str(pipeline.classes_[best_i])

        if best_prob >= THRESH:
            label = best_class
        conf = best_prob

    cvzone.putTextRect(frame, f"{label}  ({conf:.2f})", (250, 80))
    cvzone.putTextRect(frame, f"thresh={THRESH:.2f}  (+/-)", (10, 130))

    all_frames = cvzone.stackImages([real_frame, frame], cols=2, scale=0.7)
    cv2.imshow("frame", all_frames)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("+") or key == ord("="):
        THRESH = min(1.0, THRESH + 0.05)
    elif key == ord("-"):
        THRESH = max(0.0, THRESH - 0.05)
    elif key != 255:  # any key exits, like your emotion.py behavior
        break

cap.release()
cv2.destroyAllWindows()
