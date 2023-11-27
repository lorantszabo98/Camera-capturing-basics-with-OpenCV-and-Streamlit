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


def calculate_ear(eye):
    # calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    # calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])

    # calculate the EAR
    EAR = (y1 + y2) / (2.0 * x1)
    return EAR


def ear_value_calibration(calibration_ear_values, frame_count):
    if frame_count <= len(calibration_ear_values):
        mean_ear = sum(calibration_ear_values[:frame_count]) / frame_count
        return mean_ear
    else:
        return None


def set_blink_threshold(calibration_ear_values, frame_count):
    # if the calibration process was not finished (no detected face for 5 sec, then return 0)
    if len(calibration_ear_values) == 0 or frame_count == 0:
        return 0
    else:
        mean = ear_value_calibration(calibration_ear_values, frame_count)
        st.write(len(calibration_ear_values), frame_count)
        return mean - 0.05


def calculate_lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = dist.euclidean(top_mean, low_mean)
    return distance


def display_altair_chart(values, dataframe, threshold):
    column_names = list(dataframe.columns)
    st.write(column_names)
    if values:
        new_data = pd.DataFrame({column_names[0]: [len(values)], column_names[1]: [values[-1]]})
        dataframe = pd.concat([dataframe, new_data])

        # show only 50 frames EAR data at the same time
        dataframe = dataframe[-50:]

    # Create Altair Lip_distance_chart
    chart = alt.Chart(dataframe).mark_line().encode(
        x=column_names[0],
        y=column_names[1],
        tooltip=[column_names[1]]
    ).properties(width=600, height=300)

    # Add vertical line at the threshold
    threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(color='red').encode(
        y='threshold',
        size=alt.value(1)
    )

    return chart, threshold_line

# def compute_fps(ptime = 0):
#     ctime = time.time()
#     fps = int(1 / (ctime - ptime))
#     ptime = ctime
#     cv2.putText(frame, f'FPS:{fps}', (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
#                 (0, 200, 0), 3)

detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor(
    "pages/models/eye_detection/shape_predictor_68_face_landmarks_GTX.dat")


if "BLINK_TOTAL_COUNTER_" not in st.session_state:
    st.session_state.BLINK_TOTAL_COUNTER = 0

if "COUNT_FRAME" not in st.session_state:
    st.session_state.COUNT_FRAME = 0

if "YAWN_TOTAL_NUMBER" not in st.session_state:
    st.session_state.YAWN_TOTAL_NUMBER = 0


BLINK_THRES = 0.3
EYE_AR_CONSEC_FRAMES = 1

YAWN_THRES = 25


# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

cap = cv2.VideoCapture(0)

start_time = time.time()

if not cap.isOpened():
    st.error("Error: Could not open webcam.")
else:
    webcam_success_message = st.success("Webcam is active.")


info_message = st.info("Please look into the camera for 5 seconds!")
# Create a progress bar
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

ear_values = []
calibration_ear_values = []
lip_distance_values = []

# Create a DataFrame for Altair Lip_distance_chart
EAR_chart_data = pd.DataFrame(columns=['frame', 'EAR'])

LIP_chart_data = pd.DataFrame(columns=['frame', 'lip_distance'])

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        st.warning("Warning: Could not read a frame from the webcam.")
        break

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting the faces
    faces = detector(image_gray)

    # if len(faces) == 1:

    for face in faces:

        # x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # landmark detection
        shape = landmark_predict(image_gray, face)

        # converting the shape class directly
        # to a list of (x,y) coordinates
        shape = face_utils.shape_to_np(shape)

        face_hull = cv2.convexHull(shape)
        cv2.drawContours(frame, [face_hull], -1, (0, 255, 0), 1)

        # parsing the landmarks list to extract
        # lefteye and righteye landmarks--#
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

        if lip_dist > YAWN_THRES:
            st.session_state.YAWN_TOTAL_NUMBER += 1

        yawn_text.text(f" Total yawns: {st.session_state.YAWN_TOTAL_NUMBER}")

        elapsed_time = time.time() - start_time

        # calibration for dynamic EAR in the first 5 sec
        if elapsed_time <= 5:
            progress_value = min(100, int((elapsed_time / 5) * 100))  # Scale to 0-100
            progress_bar.progress(progress_value, text="Calibration in progress....")

            calibration_ear_values.append(avg)
            frame_counter += 1

        # after the first 5 sec the blink detection could start
        if elapsed_time >= 5:
            info_message.empty()
            progress_bar.empty()

            # st.write(calibration_ear_values)
            BLINK_THRES = set_blink_threshold(calibration_ear_values, frame_counter)

            ear_values.append(avg)

            if avg < BLINK_THRES:
                st.session_state.COUNT_FRAME += 1  # incrementing the frame count
            else:
                if st.session_state.COUNT_FRAME >= EYE_AR_CONSEC_FRAMES:
                    st.session_state.BLINK_TOTAL_COUNTER += 1

                st.session_state.COUNT_FRAME = 0

                blink_text.text(f"Total Blinks: {st.session_state.BLINK_TOTAL_COUNTER}")

            # altairchart for EAR display
            if ear_values:
                EAR_new_data = pd.DataFrame({'frame': [len(ear_values)], 'EAR': [ear_values[-1]]})
                EAR_chart_data = pd.concat([EAR_chart_data, EAR_new_data])

                # show only 50 frames EAR data at the same time
                EAR_chart_data = EAR_chart_data[-50:]

            # Create Altair Lip_distance_chart
            EAR_chart = alt.Chart(EAR_chart_data).mark_line().encode(
                x='frame',
                y='EAR',
                tooltip=['EAR']
            ).properties(width=600, height=300)

            # Add vertical line at the threshold
            EAR_threshold_line = alt.Chart(pd.DataFrame({'threshold': [BLINK_THRES]})).mark_rule(color='red').encode(
                y='threshold',
                size=alt.value(1)
            )

            # altairchart for lip distance display
            if lip_distance_values:
                LIP_new_data = pd.DataFrame(
                    {'frame': [len(lip_distance_values)], 'lip_distance': [lip_distance_values[-1]]})
                LIP_chart_data = pd.concat([LIP_chart_data, LIP_new_data])

                # show only 50 frames EAR data at the same time
                LIP_chart_data = LIP_chart_data[-50:]

            # Create Altair Lip_distance_chart
            LIP_chart = alt.Chart(LIP_chart_data).mark_line().encode(
                x='frame',
                y='lip_distance',
                tooltip=['lip_distance']
            ).properties(width=600, height=300)

            # Add vertical line at the threshold
            LIP_threshold_line = alt.Chart(pd.DataFrame({'threshold': [YAWN_THRES]})).mark_rule(color='red').encode(
                y='threshold',
                size=alt.value(1)
            )

            # EAR_chart, EAR_threshold_line = display_altair_chart(ear_values, EAR_chart_data, BLINK_THRES)
            # LIP_chart, LIP_threshold_line = display_altair_chart(lip_distance_values, LIP_chart_data, YAWN_THRES)

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


