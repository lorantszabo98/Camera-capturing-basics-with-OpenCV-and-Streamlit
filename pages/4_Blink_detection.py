import streamlit as st
import cv2
import dlib
import imutils
import pandas as pd
import altair as alt

from scipy.spatial import distance as dist
from imutils import face_utils
from streamlit_extras.chart_annotations import get_annotations_chart


def calculate_EAR(eye):
    # calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    # calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])

    # calculate the EAR
    EAR = (y1 + y2) / (2.0 * x1)
    return EAR

detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor(
    "pages/models/eye_detection/shape_predictor_68_face_landmarks_GTX.dat")


if "TOTAL_COUNTER" not in st.session_state:
    st.session_state.TOTAL_COUNTER = 0

if "COUNT_FRAME" not in st.session_state:
    st.session_state.COUNT_FRAME = 0


# Variables
BLINK_THRES = 0.3
EYE_AR_CONSEC_FRAMES = 1
count_frame = 0

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open webcam.")
else:
    webcam_success_message = st.success("Webcam is active.")

blink_text = st.text("Total Blinks: 0")
webcam_placeholder = st.empty()
chart_placeholder = st.line_chart([])

ear_values = []

# Create a DataFrame for Altair chart
chart_data = pd.DataFrame(columns=['frame', 'EAR'])

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        st.warning("Warning: Could not read a frame from the webcam.")
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting the faces
    faces = detector(img_gray)

    # if len(faces) == 1:

    for face in faces:

        # x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # landmark detection
        shape = landmark_predict(img_gray, face)

        # converting the shape class directly
        # to a list of (x,y) coordinates
        shape = face_utils.shape_to_np(shape)

        hull = cv2.convexHull(shape)
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

        # parsing the landmarks list to extract
        # lefteye and righteye landmarks--#
        lefteye = shape[L_start: L_end]
        righteye = shape[R_start:R_end]

        # Calculate the EAR
        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)

        # Avg of left and right eye EAR
        avg = (left_EAR + right_EAR) / 2

        leftEyeHull = cv2.convexHull(lefteye)
        rightEyeHull = cv2.convexHull(righteye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # st.toast(avg)
        ear_values.append(avg)

        if avg < BLINK_THRES:
            st.session_state.COUNT_FRAME += 1  # incrementing the frame count
        else:
            if st.session_state.COUNT_FRAME >= EYE_AR_CONSEC_FRAMES:
                # cv2.putText(frame, 'Blink Detected', (30, 30),
                #             cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
                st.session_state.TOTAL_COUNTER += 1

            st.session_state.COUNT_FRAME = 0

            blink_text.text(f"Total Blinks: {st.session_state.TOTAL_COUNTER}")

            # cv2.putText(frame, "Blinks: {}".format(st.session_state.TOTAL_COUNTER), (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # elif len(faces) < 1:
    #     st.error("No faces are detected, please upload another photo!")
    #
    # else:
    #     st.error("Multiple faces are detected, please upload another photo!")

    webcam_placeholder.image(frame, channels="BGR", use_column_width=True)

    if ear_values:
        new_data = pd.DataFrame({'frame': [len(ear_values)], 'EAR': [ear_values[-1]]})
        chart_data = pd.concat([chart_data, new_data])

    # Create Altair chart
    chart = alt.Chart(chart_data).mark_line().encode(
        x='frame',
        y='EAR',
        tooltip=['EAR']
    ).properties(width=600, height=300)

    # Add vertical line at the threshold
    threshold_line = alt.Chart(pd.DataFrame({'threshold': [BLINK_THRES]})).mark_rule(color='red').encode(
        y='threshold',
        size=alt.value(1)
    )

    chart_placeholder.altair_chart(chart + threshold_line)

    # Append the new EAR value to the chart
    # chart_placeholder.line_chart(ear_values)



# while 1:
#
#     # If the video is finished then reset it
#     # to the start
#     if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(
#             cv2.CAP_PROP_FRAME_COUNT):
#         cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
#     else:
#         _, frame = cam.read()
#         frame = imutils.resize(frame, width=640)
#
#         # converting frame to gray scale to
#         # pass to detector
#         img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # detecting the faces
#         faces = detector(img_gray)
#         for face in faces:
#
#             # landmark detection
#             shape = landmark_predict(img_gray, face)
#
#             # converting the shape class directly
#             # to a list of (x,y) coordinates
#             shape = face_utils.shape_to_np(shape)
#
#             # parsing the landmarks list to extract
#             # lefteye and righteye landmarks--#
#             lefteye = shape[L_start: L_end]
#             righteye = shape[R_start:R_end]
#
#             # Calculate the EAR
#             left_EAR = calculate_EAR(lefteye)
#             right_EAR = calculate_EAR(righteye)
#
#             # Avg of left and right eye EAR
#             avg = (left_EAR + right_EAR) / 2
#             if avg < BLINK_THRES:
#                 count_frame += 1  # incrementing the frame count
#             else:
#                 if count_frame >= EYE_AR_CONSEC_FRAMES:
#                     cv2.putText(frame, 'Blink Detected', (30, 30),
#                                 cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
#                 else:
#                     count_frame = 0
#
#         cv2.imshow("Video", frame)
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break
#
# cam.release()
# cv2.destroyAllWindows()