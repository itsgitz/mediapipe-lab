import cv2
import mediapipe as mp

def run_pose_detection():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    print("👀 Press ESC to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("❌ Failed to read frame from webcam.")
            break

        # Convert the image color (MediaPipe expects RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with MediaPipe
        results = pose.process(image_rgb)

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the result
        cv2.imshow('MediaPipe Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pose_detection()
