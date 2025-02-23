import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from predict import predict_flop, getReason 

clf = joblib.load('flop_classifier.pkl')
yolo_model = YOLO('yolov8n-pose.pt')



st.set_page_config(page_title="Soccer Foul or Dive Prediction", layout="wide")

page = st.sidebar.radio("",
                        ["🏠 Home", "🔍 How It Works"],
                        index=0,
                        key="sidebar_radio")

if page == "🏠 Home":
    st.title("Soccer Foul or Dive Prediction")
    st.write("Upload a soccer video to detect if a player committed a foul or dove.")

    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

    if uploaded_video is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        st.video(uploaded_video)

        st.write("Processing video, please wait...")

        result = predict_flop(temp_video_path)
        if result is not None:
            avg_proba, percent_diff, verdict = result
            reason = getReason(verdict.lower(), percent_diff)

            st.subheader("Prediction Results")
            st.write(f"Prediction: {verdict}")
            st.write(f"Probability of a Dive: {avg_proba * 100:.2f}%")
            st.write(f"Reason: {reason}")
        else:
            st.write("No features extracted from the video. Cannot make a prediction.")

elif page == "🔍 How It Works":
    st.title("How the Model Works")

    st.write("""
    This model works by:
    1. Using YOLO for pose estimation to track players' joint movements in the video.
    """)

    col1, col2 = st.columns(2)

    with col1:
        gif_1_path = './flop199.gif' 
        st.image(gif_1_path)
        st.caption("Original video.")

    with col2:
        gif_2_path = 'first_skeleton_output.gif' 
        st.image(gif_2_path)
        st.caption("Joint movement tracked.")

    st.write("""
    2. Extracting relevant features such as velocity, acceleration, torso angle, contact, and reaction time from temporal and spatial data about player joint movements.
    3. Using a random forest classifier (RFC) to learn common features of fouls and dives enabling predictive capabilities.
    4. Considering the most impactful features from the RFC to get an "LLM referee" to give a reason for the AI judgment.
    """)

