import streamlit as st
import cv2
import dlib
import imutils
import pandas as pd
import altair as alt
import time
import numpy as np

from scipy.spatial import distance as dist
from imutils import face_utils


# calculate EAR (eye aspect ratio) for blink detection
def calculate_ear(eye):
    # calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    # calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])

    # calculate the EAR
    EAR = (y1 + y2) / (2.0 * x1)
    return EAR


# calculate the mean of the captured EAR values during the calibration process
def ear_value_calibration(calibration_ear_values, frame_count):
    if frame_count <= len(calibration_ear_values):
        mean_ear = sum(calibration_ear_values[:frame_count]) / frame_count
        return mean_ear
    else:
        return None


# set blink threshold by dividing the calibration value by a constant
def set_blink_threshold(calibration_ear_values, frame_count):
    # if the calibration process was not finished (no detected face for 5 sec, then return 0)
    if len(calibration_ear_values) == 0 or frame_count == 0:
        return 0
    else:
        mean = ear_value_calibration(calibration_ear_values, frame_count)
        return mean - 0.1


# calculating lip distance based on facial landmarks
def calculate_lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = dist.euclidean(top_mean, low_mean)
    return distance


# display altair chart for EAR and lip distance
def display_altair_chart(values, dataframe, threshold):
    column_names = list(dataframe.columns)
    if not column_names:
        st.warning("DataFrame is empty. Provide a DataFrame with valid columns.")
        return

    x_column = column_names[0]
    y_column = column_names[1]

    if values:
        new_data = pd.DataFrame({x_column: range(1, len(values) + 1), y_column: values})
        dataframe = pd.concat([dataframe, new_data])

        # show only 50 frames EAR data at the same time
        dataframe = dataframe[-50:]

    # Create Altair Lip_distance_chart
    chart = alt.Chart(dataframe).mark_line().encode(
        x=x_column,
        y=y_column,
        tooltip=[y_column]
    ).properties(width=600, height=300)

    # Add vertical line at the threshold
    threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(color='red').encode(
        y='threshold',
        size=alt.value(1)
    )

    return chart, threshold_line


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    face_detctor = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(
        "pages/models/eye_detection/shape_predictor_68_face_landmarks_GTX.dat")

    return face_detctor, landmark_predictor

# def compute_fps(ptime = 0):
#     ctime = time.time()
#     fps = int(1 / (ctime - ptime))
#     ptime = ctime
#     cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
#                 (0, 200, 0), 3)

if "BLINK_TOTAL_COUNTER_" not in st.session_state:
    st.session_state.BLINK_TOTAL_COUNTER = 0

if "BLINK_COUNT_FRAME" not in st.session_state:
    st.session_state.BLINK_COUNT_FRAME = 0

if "YAWN_TOTAL_NUMBER" not in st.session_state:
    st.session_state.YAWN_TOTAL_NUMBER = 0

if "YAWN_COUNT_FRAME" not in st.session_state:
    st.session_state.YAWN_COUNT_FRAME = 0

# default values for blink detection
BLINK_THRESHOLD = 0.3
EYE_AR_CONSEC_FRAMES = 1

# default values for yawn detection
YAWN_THRESHOLD = 25
YAWN_AR_CONSEC_FRAMES = 10

# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

cap = cv2.VideoCapture(0)

start_time = time.time()

if not cap.isOpened():
    st.error("Error: Could not open webcam.")
else:
    webcam_success_message = st.success("Webcam is active.")

# Creating placeholders
info_message = st.info("Please prepare for the calibration process, look at your webcam picture and keep your mouth shut!")
progress_bar = st.progress(0)
webcam_placeholder = st.empty()
blink_text = st.text("Total Blinks: 0")
yawn_text = st.text("Total Yawns: 0")
# error_message = st.error()

tab1, tab2 = st.tabs(["EAR", "Lip distance"])

with tab1:
    EAR_chart_placeholder = st.line_chart([])
with tab2:
    lip_distance_chart_placeholder = st.line_chart([])

# init list for EAR values, for EAR calibration values and lip distance values
ear_values = []
calibration_ear_values = []
lip_distance_values = []

# Create a DataFrame for Altair Lip_distance_chart
EAR_chart_data = pd.DataFrame(columns=['frame', 'EAR'])
LIP_chart_data = pd.DataFrame(columns=['frame', 'lip_distance'])

# loading dlib models for face and landmark detection
face_detector, landmark_predictor = load_models()

# counting frames for the EAR calibration process
frame_counter = 0

# capturing frames in a while loop
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        st.warning("Warning: Could not read a frame from the webcam.")
        break

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting the faces
    faces = face_detector(image_gray)

    # if len(faces) == 1:

    for face in faces:

        # x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # landmark detection
        shape = landmark_predictor(image_gray, face)

        # converting the shape class directly
        # to a list of (x,y) coordinates
        shape = face_utils.shape_to_np(shape)

        face_hull = cv2.convexHull(shape)
        cv2.drawContours(frame, [face_hull], -1, (0, 255, 0), 1)

        left_eye = shape[L_start: L_end]
        right_eye = shape[R_start:R_end]

        # Calculate the EAR
        left_EAR = calculate_ear(left_eye)
        right_EAR = calculate_ear(right_eye)

        # Avg of left and right eye EAR
        avg = (left_EAR + right_EAR) / 2

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        lip_hull = cv2.convexHull(lip)
        cv2.drawContours(frame, [lip_hull], -1, (0, 255, 0), 1)

        lip_dist = calculate_lip_distance(shape)
        lip_distance_values.append(lip_dist)

        # Yawn detection
        if lip_dist < YAWN_THRESHOLD:
            st.session_state.YAWN_COUNT_FRAME += 1  # incrementing the frame count
        else:
            if st.session_state.YAWN_COUNT_FRAME >= YAWN_AR_CONSEC_FRAMES:
                st.session_state.YAWN_TOTAL_NUMBER += 1

            st.session_state.YAWN_COUNT_FRAME = 0

            yawn_text.text(f" Total yawns: {st.session_state.YAWN_TOTAL_NUMBER}")

        elapsed_time = time.time() - start_time

        # calibration for dynamic EAR in the first 5 sec
        if 5 <= elapsed_time <= 10:
            progress_value = min(100, int((elapsed_time / 10) * 100))
            progress_bar.progress(progress_value, text="Calibration in progress....")
            info_message.info("Please look at your webcam picture and keep your mouth shut for 5 seconds!")

            calibration_ear_values.append(avg)
            frame_counter += 1

        # after the first 5 sec the blink detection could start
        if elapsed_time >= 10:
            info_message.empty()
            progress_bar.empty()

            BLINK_THRESHOLD = set_blink_threshold(calibration_ear_values, frame_counter)

            ear_values.append(avg)

            # blink detection
            if avg < BLINK_THRESHOLD:
                st.session_state.BLINK_COUNT_FRAME += 1  # incrementing the frame count
            else:
                if st.session_state.BLINK_COUNT_FRAME >= EYE_AR_CONSEC_FRAMES:
                    st.session_state.BLINK_TOTAL_COUNTER += 1

                st.session_state.BLINK_COUNT_FRAME = 0

                blink_text.text(f"Total Blinks: {st.session_state.BLINK_TOTAL_COUNTER}")

            # displaying the Altair charts
            EAR_chart, EAR_threshold_line = display_altair_chart(ear_values, EAR_chart_data, BLINK_THRESHOLD)
            LIP_chart, LIP_threshold_line = display_altair_chart(lip_distance_values, LIP_chart_data, YAWN_THRESHOLD)

            with tab1:
                EAR_chart_placeholder.altair_chart(EAR_chart + EAR_threshold_line)

            with tab2:
                lip_distance_chart_placeholder.altair_chart(LIP_chart + LIP_threshold_line)

    # elif len(faces) < 1:
    #     st.error("No faces are detected, please upload another photo!")
    #
    # else:
    #     st.error("Multiple faces are detected, please upload another photo!")

    webcam_placeholder.image(frame, channels="BGR", use_column_width=True)


