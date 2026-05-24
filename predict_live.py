import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load trained model
model = joblib.load("sign_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

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

            # Extract landmarks
            row = []
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]

            row = np.array(row).reshape(1, -1)

            prediction = model.predict(row)[0]

            cv2.putText(frame, f"Prediction: {prediction}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 255, 0), 3)

    cv2.imshow("Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
