import cv2
import mediapipe as mp
import pandas as pd
import signal
import sys

# Initialize MediaPipe Holistic, Drawing Utilities, and Styles
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Define a handler for keyboard interrupt to ensure clean exit
def signal_handler(sig, frame):
    print('Exiting gracefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def create_frame_landmark_df(results, frame, xyz):
    xyz_skel = xyz[['type', 'landmark_index']].drop_duplicates().reset_index(drop=True).copy()

    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()

    if results.face_landmarks is not None:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.pose_landmarks is not None:
        for i, point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks is not None:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks is not None:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
            
    face = face.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='face')
    pose = pose.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='pose')
    left_hand = left_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='left_hand')
    right_hand = right_hand.reset_index().rename(columns={'index': 'landmark_index'}).assign(type='right_hand')

    landmarks = pd.concat([face, pose, left_hand, right_hand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=['type', 'landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def do_capture_loop(xyz):
    all_landmarks = []
    try:
        # Open webcam
        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            frame = 0
            while True:
                frame += 1
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Process the image and extract landmarks
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                # Create landmark dataframe
                landmarks = create_frame_landmark_df(results, frame, xyz)
                all_landmarks.append(landmarks)

                # Draw landmarks on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # Flip the image horizontally for a selfie-view display
                cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

                # Save landmarks to Parquet file
                landmarks_df = pd.concat(all_landmarks).reset_index(drop=True)
                landmarks_df.to_parquet('output.parquet')

                # Exit loop when ESC key is pressed
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pq_file = './input/asl-signs/train_landmark_files/2044/635217.parquet'
    xyz = pd.read_parquet(pq_file)
    do_capture_loop(xyz)
