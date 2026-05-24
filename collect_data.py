import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

dataset_file = "dataset.csv"

# Create CSV header if not exists
if not os.path.exists(dataset_file):
    with open(dataset_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

current_label = None
collecting = False

print("Press A/B/C... to collect that letter")
print("Press S to stop collecting")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if collecting and current_label:
                row = []
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                row.append(current_label)

                with open(dataset_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                cv2.putText(frame, f"Collecting: {current_label}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        collecting = False
        current_label = None
    elif chr(key).isalpha():
        current_label = chr(key).upper()
        collecting = True

cap.release()
cv2.destroyAllWindows()
