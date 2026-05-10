import cv2
import numpy as np
import mediapipe as mp

print("🚀 Air Whiteboard v1 Started")

# ---------------- Mediapipe Setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

# ---------------- Main Loop ----------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Create canvas
    if canvas is None:
        canvas = np.zeros((h, w), dtype=np.uint8)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        lm = hand_landmarks.landmark

        # Index fingertip coordinates
        x = int(lm[8].x * w)
        y = int(lm[8].y * h)

        # Finger states
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_up = lm[16].y < lm[14].y
        pinky_up = lm[20].y < lm[18].y

        # ---------------- DRAW MODE ----------------
        # Only index finger up
        if index_up and not middle_up:

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 5)

            prev_x, prev_y = x, y

            cv2.putText(frame, "DRAW MODE", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        # ---------------- ERASER MODE ----------------
        # Palm open = all fingers up
        elif index_up and middle_up and ring_up and pinky_up:

            cv2.circle(canvas, (x, y), 30, 0, -1)

            prev_x, prev_y = 0, 0

            cv2.putText(frame, "ERASER MODE", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        else:
            prev_x, prev_y = 0, 0

        # Cursor
        cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)

    # ---------------- Overlay Canvas ----------------
    output = cv2.add(frame, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    # Show window
    cv2.imshow("Air Whiteboard v1", output)

    key = cv2.waitKey(1)

    # ---------------- Keyboard Controls ----------------

    # Clear canvas
    if key == ord('c'):
        canvas = np.zeros((h, w), dtype=np.uint8)
        print("🧹 Canvas Cleared")

    # Quit
    elif key == ord('q'):
        break

# ---------------- Cleanup ----------------
cap.release()
cv2.destroyAllWindows()
