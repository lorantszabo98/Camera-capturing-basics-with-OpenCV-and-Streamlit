import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
import dlib

from Data_manager import add_property_to_session


@st.cache_resource(show_spinner="Please wait...")
def models_loading(model):
    return cv2.dnn.readNetFromCaffe(f"pages/models/{model}/{model}_deploy.prototxt",
                                    f"pages/models/{model}/{model}_net.caffemodel")


def detect_label(frame, model, labels, attribute_name):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    model.setInput(blob)
    prediction = model.forward()
    predicted_class = np.argmax(prediction)
    detection = labels[predicted_class]

    st.session_state.user_info[attribute_name] = detection

    return detection


def detection(frame, detection_type):
    if detection_type not in models or detection_type not in labels:
        st.warning(f"Unsupported detection type: {detection_type}")
        return None

    model = models[detection_type]
    label = labels[detection_type]

    return detect_label(frame, model, label, detection_type)


def disappearing_success_message(message_text, sleeptime):
    message = st.success(message_text)
    time.sleep(sleeptime)
    message.empty()


if "user_info" not in st.session_state:
    st.session_state.user_info = {
        "Name": "",
        "Date": datetime.now().date(),
        "Gender": "Unknown",
        "Age": "Unknown"
    }

if "start_webcam" not in st.session_state:
    st.session_state.start_webcam = False

models = {
    "Gender": models_loading("gender"),
    "Age": models_loading("age"),
    # Add other detection types if needed
}

labels = {
    "Gender": ["Male", "Female"],
    "Age": ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)'],
    # Add other detection types if needed
}

detector = dlib.get_frontal_face_detector()

st.title("Webcam Capture")

user_name = st.text_input("Please enter your name")
current_date = st.session_state.user_info["Date"]

if st.button("Submit Name"):
    if not user_name:
        st.error("Please enter a name!")
    else:
        st.session_state.user_info["Name"] = user_name
        st.session_state.user_info["Gender"] = "Unknown"
        st.session_state.user_info["Age"] = "Unknown"
        disappearing_success_message("Entry added Successfully", 2)

        st.session_state.start_webcam = True

# start_webcam = st.checkbox("Start Webcam", key="webcam_start")

if st.session_state.start_webcam:

    # Create a VideoCapture object for the webcam (0 for the default webcam)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        webcam_success_message = st.success("Webcam is active.")

    selectbox_option = st.sidebar.selectbox(
        "Choose what to detect on you",
        ("None", "Gender", "Age"),
        key="detection option"
    )

    if st.sidebar.button("Add user data to the database"):
        entry = {
            "Name": st.session_state.user_info["Name"],
            "Date": st.session_state.user_info["Date"],
            "Gender": st.session_state.user_info["Gender"],
            "Age": st.session_state.user_info["Age"]
        }
        add_property_to_session(entry)
        with st.sidebar:
            disappearing_success_message("User data added successfully!", 2)
        st.session_state.start_webcam = False

    # Video recording for later
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  # Adjust parameters as needed

    # Create a placeholder for webcam feed and detection results
    webcam_placeholder = st.empty()

    # Initialize the previous detection result
    previous_detection_result = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.warning("Warning: Could not read a frame from the webcam.")
            break

        faces = detector(frame)

        if len(faces) < 1:
            with st.sidebar:
                st.toast("No faces are detected!")
                webcam_placeholder.image(frame, channels="BGR", use_column_width=True)
        elif len(faces) > 1:
            st.toast("Multiple faces are detected!")
        else:
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if selectbox_option == "Gender":
                current_detection_result = detection(frame, "Gender")
                pass
            elif selectbox_option == "Age":
                current_detection_result = detection(frame, "Age")
                pass

            else:
                current_detection_result = None

                # Update the placeholder with the current webcam frame
            if current_detection_result:
                # Draw the detection result text on the frame using cv2.putText
                cv2.putText(frame, current_detection_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Update the placeholder with the current webcam frame
            webcam_placeholder.image(frame, channels="BGR", use_column_width=True)

            # Display the frame in the Streamlit app
            # placeholder.image(frame, channels="BGR", use_column_width=True)

            # out.write(frame)

    # Release the webcam when done
    cap.release()
    # out.release()
