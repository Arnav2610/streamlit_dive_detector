import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import joblib
import json
import google.generativeai as genai

genai.configure(api_key=os.getenv("AIzaSyAjwSb3D1loKnHrlb6NxzjcnC1nSHjOCCk")) 

genai_model = genai.GenerativeModel("gemini-1.5-flash")

def getReason(verdict, weights):
    if verdict == "foul":
        custom_prompt = (
            "I have data about a soccer player's joint movement. "
            "According to my ML random forest classifier with the following parameters: y velocity, acceleration, torso angle, contact, reaction time. "
            f"The feature differences are: {weights}. "
            "This player was fouled. "
            "Tell me the biggest reason there was a foul here. For example, a collision between the player's legs or nearly instant reaction time. "
            "Respond like a soccer referee without mentioning the model or the weights."
            "Also, dont talk about the torso angle."
        )
    else:
        custom_prompt = (
            "I have data about a soccer player's joint movement. "
            "According to my ML random forest classifier with the following parameters: y velocity, acceleration, torso angle, contact, reaction time. "
            f"The feature differences are: {weights}. "
            "This player flopped (was not actually fouled). "
            "Tell me the biggest reason there wasn't a foul here. For example, no contact between the player's legs or exaggerated fall timing. "
            "Respond like a soccer referee without mentioning the model or the weights."
            "Also, dont talk about the torso angle."
        )
    try:
        response = genai_model.generate_content(custom_prompt)
        if not response.text:
            return "No explanation provided."
        return response.text.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"
    
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

baseline = {
    'velocity': 0.01912354192667923,
    'acceleration': 0.9464381441927009,
    'torso_angle': 0.034118353059798746,
    'contact': 0.0002936621262937371,
    'reaction_time': 2.6298694527314527e-05
}

THRESHOLD = 0.5

yolo_model = YOLO('yolov8n-pose.pt')

def extract_features(video_path):
    video = cv2.VideoCapture(video_path)
    previous_keypoints = None
    previous_time = None
    features = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as pbar:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            results = yolo_model.predict(frame, conf=0.3, show=False, verbose=False)
            current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                current_skeletons = []
                for skeleton in results[0].keypoints.xy:
                    skel = skeleton.tolist()
                    if len(skel) < 13:
                        continue
                    current_skeletons.append(skel)
                
                if previous_keypoints is not None and current_skeletons:
                    dt = current_time - previous_time
                    if dt > 0:
                        num_skel = min(len(previous_keypoints), len(current_skeletons))
                        for i in range(num_skel):
                            prev_skel = previous_keypoints[i]
                            curr_skel = current_skeletons[i]
                            velocities = [(curr_skel[j][1] - prev_skel[j][1]) / dt for j in range(len(curr_skel))]
                            accelerations = [(velocities[j] - ((prev_skel[j][1] - curr_skel[j][1]) / dt)) / dt for j in range(len(curr_skel))]
                            shoulder = curr_skel[5]
                            hip = curr_skel[11]
                            torso_angle = np.arctan2(hip[1] - shoulder[1], hip[0] - shoulder[0]) * (180 / np.pi)
                            contact_detected = (euclidean_distance(curr_skel[5], curr_skel[6]) < 50 or
                                                euclidean_distance(curr_skel[11], curr_skel[12]) < 50)
                            reaction_time = dt
                            
                            feature_vector = [
                                np.mean(velocities),
                                np.mean(accelerations),
                                torso_angle,
                                int(contact_detected),
                                reaction_time
                            ]
                            features.append(feature_vector)
                
                previous_keypoints = current_skeletons
                previous_time = current_time
            pbar.update(1)
    
    video.release()
    return features

def predict_flop(video_path):
    features = extract_features(video_path)
    if not features:
        print("No features extracted from the video. Cannot make a prediction.")
        return None
    
    clf = joblib.load('flop_classifier.pkl')
    feature_columns = ['velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time']
    df_features = pd.DataFrame(features, columns=feature_columns)
    
    probas = clf.predict_proba(df_features)
    avg_proba = np.mean(probas[:, 1])
    
    avg_feature_values = np.mean(df_features.values, axis=0)
    norm_feature_values = avg_feature_values / np.sum(avg_feature_values)
    
    percent_diff = {}
    for i, feature in enumerate(feature_columns):
        base_val = baseline[feature]
        current_val = norm_feature_values[i]
        percent_difference = ((current_val - base_val) / base_val) * 100
        percent_diff[feature] = percent_difference
    
    final_verdict = "Flop" if avg_proba > THRESHOLD else "Foul"
    return avg_proba, percent_diff, final_verdict

if __name__ == '__main__':
    test_folder = 'test'
    valid_exts = ('.mp4', '.mov', '.avi')
    test_files = [f for f in os.listdir(test_folder)
                  if os.path.isfile(os.path.join(test_folder, f)) and f.lower().endswith(valid_exts)]
    test_files = sorted(set(test_files))
    
    for file in test_files:
        video_path = os.path.join(test_folder, file)
        result = predict_flop(video_path)
        if result is not None:
            avg_proba, percent_diff, verdict = result
            reason = getReason(verdict.lower(), percent_diff)
            print(f"File: {file}, Prediction: {verdict}, Probability: {avg_proba:.2f}, Features: {percent_diff}")
            print(f"Reason: {reason}\n")
    

