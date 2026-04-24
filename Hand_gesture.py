import cv2
import mediapipe as mp
import numpy as np

# --------- Load Hand Landmarker (NEW API) ---------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# You need a model file (download link below)
model_path = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# --------- Webcam ---------
cap = cv2.VideoCapture(0)

def count_fingers(landmarks):
    fingers = []

    # Thumb (simple logic)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    tips = [8, 12, 16, 20]
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


# --------- Main Loop ---------
frame_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Process frame
    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    gesture = "No Hand"

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            landmarks = hand_landmarks

            # Draw landmarks
            for lm in landmarks:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            fingers = count_fingers(landmarks)
            total = sum(fingers)

            # Gesture logic
            if total == 0:
                gesture = "Fist ✊"
            elif total == 5:
                gesture = "Open Hand ✋"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture = "Thumbs Up 👍"
            else:
                gesture = f"Fingers: {total}"

    # Display
    cv2.putText(frame, gesture, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition (New API)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()