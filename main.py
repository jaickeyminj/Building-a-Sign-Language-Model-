from flask import Flask, render_template, Response
import cv2
import os
import mediapipe as mp
import pandas as pd
from pathlib import Path
import uuid 
import tensorflow as tf
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path

unique_id = str(uuid.uuid4().hex)[:8]

def delete_files_in_folder(folder_path):
    try:
        # Get the list of files in the folder
        files = os.listdir(folder_path)

        # Iterate over each file and delete
        for file in files:
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        print(f"All files in {folder_path} deleted successfully.")
    except Exception as e:
        print(f"Error deleting files: {e}")

# Specify the folder path
folder_path = "E:/Machine Learning DSL505/ASG1/land"
delete_files_in_folder(folder_path)
folder_path = "E:/Machine Learning DSL505/ASG1/tf"
delete_files_in_folder(folder_path)
app = Flask(__name__)

# Initialize MediaPipe Hands and Holistic (for pose landmarks)
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
hands = mp_hands.Hands()
holistic = mp_holistic.Holistic()

# Initialize OpenCV
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# List to store the landmark values
landmark_data = []

def generate_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands and Holistic
        hand_results = hands.process(rgb_frame)
        holistic_results = holistic.process(rgb_frame)

        # Your existing code for extracting landmarks and displaying the frame
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Get the handedness (left or right)
                handedness = hand_results.multi_handedness[0].classification[0].label

                # Dictionary to store hand landmark values for this frame
                hand_landmarks_dict = {}

                # Store all hand landmarks
                for i, landmark in enumerate(hand_landmarks.landmark):
                    hand_landmarks_dict[f"x_{handedness.lower()}_hand_{i}"] = landmark.x
                    hand_landmarks_dict[f"y_{handedness.lower()}_hand_{i}"] = landmark.y
                    hand_landmarks_dict[f"z_{handedness.lower()}_hand_{i}"] = landmark.z

                # Append the hand frame's landmarks to the list
                landmark_data.append(hand_landmarks_dict)

        # Check if pose landmarks are detected
        if holistic_results.pose_landmarks:
            # Dictionary to store pose landmark values for this frame
            pose_landmarks_dict = {}

            # Store all pose landmarks
            for i, landmark in enumerate(holistic_results.pose_landmarks.landmark):
                pose_landmarks_dict[f"x_pose_{i}"] = landmark.x
                pose_landmarks_dict[f"y_pose_{i}"] = landmark.y
                pose_landmarks_dict[f"z_pose_{i}"] = landmark.z

            # Append the pose frame's landmarks to the list
            landmark_data.append(pose_landmarks_dict)

        # Display the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        df = pd.DataFrame(landmark_data)
        unique_id = str(uuid.uuid4().hex)[:8]
        # Save the DataFrame to a Parquet file
        parquet_file = Path(f"land/landmark_data_{unique_id}.parquet")
        df.to_parquet(parquet_file, engine="pyarrow")

        file_path = f"landmark_data.parquet"
        df = pd.read_parquet(parquet_file)
        # Pose coordinates for hand movement.
        LPOSE = [13, 15, 17, 19, 21]
        RPOSE = [14, 16, 18, 20, 22]
        POSE = LPOSE + RPOSE
        X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
        Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
        Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]
        FEATURE_COLUMNS = X + Y + Z
        X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
        Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
        Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]

        RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "right" in col]
        LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "left" in col]
        RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
        LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]

        # Path(f"land/landmark_data_{unique_id}.parquet")
        tfrecord_file = f"tf/landmark_data_{unique_id}.tfrecord"

        # Define a function to create a TFRecord feature
        def _float_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        # Create a TFRecord writer
        with tf.io.TFRecordWriter(tfrecord_file) as writer:
            # Iterate through rows in the DataFrame
            for _, row in df.iterrows():
                # Create a feature dictionary
                feature = {
                    key: _float_feature([value]) for key, value in row.items()
                }

                # Create an Example object
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize the Example and write it to the TFRecord file
                writer.write(example.SerializeToString())


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global cap, video_stream_active
    video_stream_active = False
    cap.release()
    cv2.destroyAllWindows()
    return 'Video stream stopped!'


if __name__ == '__main__':
    app.run(debug=True)
