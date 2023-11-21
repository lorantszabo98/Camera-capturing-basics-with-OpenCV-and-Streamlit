import streamlit
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import dlib
import time

DETECTOR_MODELS = {
    "Face": dlib.get_frontal_face_detector(),
    "Eye": dlib.shape_predictor("pages/models/eye_detection/shape_predictor_68_face_landmarks_GTX.dat")
}


@st.cache_resource
def load_model(model_type):
    return DETECTOR_MODELS.get(model_type)


@st.cache_data(show_spinner="Converting image...")
def image_to_numpy(image):
    uploaded_image = Image.open(image)
    uploaded_image = ImageOps.exif_transpose(uploaded_image)

    _image_numpy_array = np.array(uploaded_image)

    return _image_numpy_array


# @st.cache_data(show_spinner="Detecting faces...")
def face_detection(_image):
    _face_detector = load_model("Face")
    _faces = _face_detector(_image)

    return _faces


def eye_detection(_image, _faces, eye="Left"):
    _face = faces[0]
    _eye_detector = load_model("Eye")
    _landmarks = _eye_detector(_image, _face)

    if eye == "Left":
        eye_landmarks = _landmarks.parts()[36:42]

    else:
        eye_landmarks = _landmarks.parts()[42:48]

    # Define the coordinates of the ROI around the left eye
    eye_x = min(landmark.x for landmark in eye_landmarks)
    eye_y = min(landmark.y for landmark in eye_landmarks)
    eye_width = max(landmark.x for landmark in eye_landmarks) - eye_x
    eye_height = max(landmark.y for landmark in eye_landmarks) - eye_y

    eye_roi = _image[eye_y:eye_y + eye_height, eye_x:eye_x + eye_width]

    return eye_roi


def image_resolution_checker(image):
    if image.shape[0] < 1000 and image.shape[1] < 1000:
        st.warning(
            "The picture resolution is too small to properly detect eyes, if you want good results please use an other photo!")


# # Initialize session state
# if 'processed_image' not in st.session_state:
#     st.session_state.processed_image = None
#
# if 'faces' not in st.session_state:
#     st.session_state.faces = None
#
# if "image" not in st.session_state:
#     st.session_state.image = None

# if "eye_detection" not in st.session_state:
#     st.session_state.eye_etection = False

st.title("Eye detection")

st.info("For eye detection please use your webcam or upload a photo of your face!")

selectbox_option = st.selectbox(
    "Which method do you want to use?",
    ("Upload Photo", "Take Photo"),
    key="detection option",
    index=0
)

uploaded_file = None

if selectbox_option == "Upload Photo":

    uploaded_file = st.file_uploader("Please upload a photo of your face", accept_multiple_files=False,
                                     type=["jpg", "jpeg", "png"])

    if uploaded_file:

        image_numpy_array = image_to_numpy(uploaded_file)

        image_resolution_checker(image_numpy_array)

        faces = face_detection(image_numpy_array)

        face_found = False
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(image_numpy_array, (x, y), (x + w, y + h), (255, 255, 0), 3)

        if len(faces) == 1:
            st.success("Success, the face detected! You can perform eye detection, if you scroll down!")
            face_found = True

        elif len(faces) < 1:
            st.error("No faces are detected, please upload another photo!")

        else:
            st.error("Multiple faces are detected, please upload another photo!")

        st.image(image_numpy_array, caption="Uploaded Photo with Faces", use_column_width=True)

        if face_found:
            perform_eye_detection = st.button("Perform eye detection")

            if perform_eye_detection:
                col1, col2 = st.columns(2)

                left_eye_roi = eye_detection(image_numpy_array, faces, "Left")
                right_eye_roi = eye_detection(image_numpy_array, faces, "Right")

                with col1:
                    st.image(left_eye_roi)
                    st.caption("The left eye")

                with col2:
                    st.image(right_eye_roi)
                    st.caption("The right eye")

            # else:
            #     st.write("Something went wrong!")

elif selectbox_option == "Take Photo":

    camera_input = st.camera_input("Take a picture of your face")

    if camera_input is not None:
        image_numpy_array = image_to_numpy(camera_input)

        image_resolution_checker(image_numpy_array)

        faces = face_detection(image_numpy_array)

        face_found = False
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(image_numpy_array, (x, y), (x + w, y + h), (255, 255, 0), 2)

        if len(faces) == 1:
            st.success("Success, the face detected! You can perform eye detection, if you scroll down!")
            face_found = True

        elif len(faces) < 1:
            st.error("No faces are detected, please take another photo!")

        else:
            st.error("Multiple faces are detected, please take another photo!")

        st.image(image_numpy_array, caption="Uploaded Photo with Faces", use_column_width=True)

        if face_found:
            perform_eye_detection = st.button("Perform eye detection")

            if perform_eye_detection:
                col1, col2 = st.columns(2)

                left_eye_roi = eye_detection(image_numpy_array, faces, "Left")
                right_eye_roi = eye_detection(image_numpy_array, faces, "Right")

                with col1:
                    st.image(left_eye_roi)
                    st.caption("The left eye")

                with col2:
                    st.image(right_eye_roi)
                    st.caption("The right eye")
