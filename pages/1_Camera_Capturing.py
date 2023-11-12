import streamlit as st
import cv2
import numpy as np
from datetime import datetime
from Data_manager import add_property_to_session
import time


@st.cache_resource
def models_loading(model):
    if model == "gender":
        # pretrained age detection model - prototxt is the network architecture, caffemodel is the weights
        selected_model = cv2.dnn.readNetFromCaffe("pages/models/gender/gender_deploy.prototxt",
                                                  "pages/models/gender/gender_net.caffemodel")
    if model == "age":
        selected_model = cv2.dnn.readNetFromCaffe("pages/models/age/age_deploy.prototxt", "pages/models/age/age_net.caffemodel")
    # if model == "emotions":
    #
    # else:

    return selected_model


# st.set_page_config(layout="wide")

st.title("Webcam Capture")

# gender_net = cv2.dnn.readNetFromCaffe("pages/models/gender/gender_deploy.prototxt", "pages/models/gender/gender_net.caffemodel")

if "user_info" not in st.session_state:
    st.session_state.user_info = {
        "Name": "",
        "Date": datetime.now().date(),
        "Gender": "Unknown",
        "Age": "Unknown"

    }

user_name = st.text_input("Please enter your name")
current_date = st.session_state.user_info["Date"]
# st.session_state.webcam_start = False

if st.button("Submit Name"):
    if not user_name:
        st.error("Please enter a name!")
    else:
        st.session_state.user_info["Name"] = user_name
        st.session_state.user_info["Gender"] = "Unknown"
        st.session_state.user_info["Age"] = "Unknown"
        entry_success_message = st.success("Entry added Successfully")
        time.sleep(2)
        entry_success_message.empty()

        st.session_state.webcam_start = True

start_webcam = st.checkbox("Start Webcam", key="webcam_start")


if st.session_state.webcam_start:

    # Create a VideoCapture object for the webcam (0 for the default webcam)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        webcam_success_message = st.success("Webcam is active.")
        # time.sleep(2)
        # webcam_success_message.empty()

    # # Create a radio button for selecting what to detect
    # detection_option = st.radio("Choose what to detect on you", ["None", "Gender", "Age", "Facial Expressions"],
    #                             key="detection_option")

    selectbox_option = st.sidebar.selectbox(
        "Choose what to detect on you",
        ("None", "Gender", "Age"),
        key="detection option"
    )

    if st.button("Add user data", key="add_user_data"):
        entry = {
            "Name": st.session_state.user_info["Name"],
            "Date": st.session_state.user_info["Date"],
            "Gender": st.session_state.user_info["Gender"],
            "Age": st.session_state.user_info["Age"]
        }
        add_property_to_session(entry)

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

        # It is used to transform the inout image, to fit the model requirements
        blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False)

        # Apply the selected detection task to the webcam frame
        if selectbox_option == "Gender":

            gender_net = models_loading("gender")

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()

            predicted_class = np.argmax(gender_preds)
            gender_labels = ["Male", "Female"]
            gender = gender_labels[predicted_class]

            # Display the frame with the detected gender
            text = f"Gender: {gender}"
            # cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # placeholder.image(frame, channels="BGR", use_column_width=True)

            st.session_state.user_info["Gender"] = gender

            current_detection_result = gender

            pass
        elif selectbox_option == "Age":

            age_net = models_loading("age")

            age_net.setInput(blob)
            age_preds = age_net.forward()

            predicted_class = np.argmax(age_preds)
            age_labels = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            age = age_labels[predicted_class]
            text = f"Age: {age}"
            # cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # placeholder.image(frame, channels="BGR", use_column_width=True)

            st.session_state.user_info["Age"] = age

            current_detection_result = age

            pass
        # elif detection_option == "Facial Expressions":
        #
        #     pass

        else:
            current_detection_result = None

            # Update the placeholder with the current webcam frame
        if current_detection_result:
            # Draw the detection result text on the frame using cv2.putText
            cv2.putText(frame, current_detection_result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update the placeholder with the current webcam frame
        webcam_placeholder.image(frame, channels="BGR", use_column_width=True)

        # If the detection result has changed, update the placeholder
        # if current_detection_result != previous_detection_result:
        #     # Update the placeholder with the current detection result
        #     # ... (add your logic here based on the current_detection_result)
        #     previous_detection_result = current_detection_result
        # # If "None" is selected, do nothing (original frame)

        # Display the frame in the Streamlit app
        # placeholder.image(frame, channels="BGR", use_column_width=True)

        # out.write(frame)

    # Release the webcam when done
    cap.release()
    # out.release()










