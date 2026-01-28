import cv2
import mediapipe as mp
import numpy as np
import csv
import os


videopath = 0   # "a.mp4"/"b.mp4"/"c.mp4" OR 0 for webcam
class_name = "c"      # used in MP4 mode (auto-label)
USE_KEY_LABEL = False # True for webcam collection

OUT_CSV = "asl_abc_data.csv"

# Header (run once then comment out, like your style)

# Columns = ["Class"]
# for i in range(1, 22):
#     Columns += [f"hand_x{i}", f"hand_y{i}", f"hand_z{i}"]
# print("NUM COLS:", len(Columns))
# with open(OUT_CSV, "w", newline="") as f:
#     csv.writer(f, delimiter=",").writerow(Columns)

# If you forgot to create the file/header, auto-create it safely:
if not os.path.exists(OUT_CSV):
    Columns = ["Class"]
    for i in range(1, 22):
        Columns += [f"hand_x{i}", f"hand_y{i}", f"hand_z{i}"]
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f, delimiter=",").writerow(Columns)

cap = cv2.VideoCapture(videopath)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,                # ONE hand only
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def get_hand_vec(result):
    """63 features: x,y,z for 21 landmarks. Returns None if no hand."""
    if result.multi_hand_landmarks:
        hlm = result.multi_hand_landmarks[0]
        coords = []
        for lm in hlm.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        return coords
    return None

while cap.isOpened():
    rt, frame = cap.read()
    if not rt:
        break

    frame = cv2.resize(frame, (720, 480))
    real_frame = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # draw hand (feedback)
    if result.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    # webcam key labeling (optional)
    if USE_KEY_LABEL:
        cv2.putText(frame, "Keys: a b c u (unknown) | q quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Label: {class_name}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    hand_vec = get_hand_vec(result)

    # Write row (like test_facial.py), but only if we have a real hand
    if hand_vec is not None:
        row = [class_name] + hand_vec
        with open(OUT_CSV, "a", newline="") as f:
            csv.writer(f, delimiter=",").writerow(row)

    # show
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if USE_KEY_LABEL:
        if key == ord("a"): class_name = "a"
        elif key == ord("b"): class_name = "b"
        elif key == ord("c"): class_name = "c"
        elif key == ord("u"): class_name = "unknown"
        elif key == ord("q"): break
    else:
        # MP4 mode: any key stops (same vibe as your original)
        if key != 255:
            break

cap.release()
cv2.destroyAllWindows()
