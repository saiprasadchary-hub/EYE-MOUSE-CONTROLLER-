import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Blink detection variables
last_blink_time = 0
blink_count = 0
blink_threshold = 0.004  # Adjust if needed
blink_in_progress = False  # Prevent multiple detections on same blink

# Eye movement smoothing
last_eye_center = (0, 0)

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Eye center for mouse movement
        eye_landmarks = [landmarks[474], landmarks[475], landmarks[476], landmarks[477]]
        eye_center_x = sum([landmark.x for landmark in eye_landmarks]) / len(eye_landmarks)
        eye_center_y = sum([landmark.y for landmark in eye_landmarks]) / len(eye_landmarks)

        screen_x = int(screen_w * eye_center_x)
        screen_y = int(screen_h * eye_center_y)

        if abs(screen_x - last_eye_center[0]) > 10 or abs(screen_y - last_eye_center[1]) > 10:
            pyautogui.moveTo(screen_x, screen_y, duration=0.01)
            last_eye_center = (screen_x, screen_y)

        # --- Eye landmarks for blinking ---

        # Left Eye (visual + blink detection)
        left_eye = [landmarks[145], landmarks[159]]
        left_eye_diff = abs(left_eye[0].y - left_eye[1].y)
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

        # Right Eye (visual only)
        right_eye = [landmarks[374], landmarks[386]]
        for landmark in right_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

        # --- Blink detection with debounce ---
        if left_eye_diff < blink_threshold and not blink_in_progress:
            blink_in_progress = True
            current_time = time.time()

            if current_time - last_blink_time < 1.2:  # Within 1.2 seconds
                blink_count += 1
            else:
                blink_count = 1  # Reset count

            last_blink_time = current_time

            if blink_count == 1:
                pyautogui.click()
                print("Left Click 👁️")
                time.sleep(0.3)

            elif blink_count == 2:
                pyautogui.rightClick()
                print("Right Click 👆")
                time.sleep(0.3)

        elif left_eye_diff >= blink_threshold:
            blink_in_progress = False  # Reset only when eye is open again

    # Show the frame
    cv2.imshow('Eye Controlled Mouse', frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
