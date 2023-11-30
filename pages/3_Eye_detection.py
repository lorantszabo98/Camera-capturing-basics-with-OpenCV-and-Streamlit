import streamlit
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist


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


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    c = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    EAR = (a + b) / (2.0 * c)

    # return the eye aspect ratio
    return EAR


# def image_resolution_checker(image):
#     if image.shape[0] < 1000 and image.shape[1] < 1000:
#         st.warning(
#             "The picture resolution is too small to properly detect eyes, if you want good results please use an other photo!")


st.title("Eye detection")
st.info("For eye detection please use your webcam or upload a photo of your face!")
selectbox_option = st.selectbox(
    "Which method do you want to use?",
    ("Upload Photo", "Take Photo"),
    key="detection option",
    index=0
)

perform_eye_detection = st.empty()

landmark_predictor = dlib.shape_predictor(
        "pages/models/eye_detection/shape_predictor_68_face_landmarks_GTX.dat")

EAR_THRESHOLD = 0.15
EAR_THRESHOLD_WEBCAM = 0.25

(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

if selectbox_option == "Upload Photo":

    uploaded_file = st.file_uploader("Please upload a photo of your face", accept_multiple_files=False,
                                     type=["jpg", "jpeg", "png"])

    progress_bar = st.empty()

    if uploaded_file:

        progress_bar.progress(0, text="Analyzing photo...")

        image = image_to_numpy(uploaded_file)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image_resolution_checker(image_gray)

        faces = face_detection(image_gray)

        face_found = False
        for face in faces:
            # x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 3)
            shape = landmark_predictor(image_gray, face)
            shape = face_utils.shape_to_np(shape)

            face_hull = cv2.convexHull(shape)
            cv2.drawContours(image, [face_hull], -1, (0, 255, 0), 2)

        message = st.empty()

        if len(faces) == 1:
            message.success("Success, the face detected! You can perform eye detection!")
            face_found = True
            perform_eye_detection = st.button("Perform eye detection")

        elif len(faces) < 1:
            message.error("No faces are detected, please upload another photo!")

        else:
            message.error("Multiple faces are detected, please upload another photo!")

        image_placeholder = st.image(image, caption="Uploaded Photo with Faces", use_column_width=True)
        progress_bar.progress(100, text="Photo analyzed!")

        if face_found:

            if perform_eye_detection:

                progress_bar.progress(0, text="Detecting eyes....")

                col1, col2 = st.columns(2)

                for face in faces:

                    shape = landmark_predictor(image_gray, face)

                    shape = face_utils.shape_to_np(shape)

                    left_eye = shape[L_start: L_end]
                    right_eye = shape[R_start:R_end]

                    left_eye_hull = cv2.convexHull(left_eye)
                    right_eye_hull = cv2.convexHull(right_eye)
                    cv2.drawContours(image, [left_eye_hull], -1, (0, 255, 0), 2)
                    cv2.drawContours(image, [right_eye_hull], -1, (0, 255, 0), 2)

                    left_EAR = eye_aspect_ratio(left_eye)
                    right_EAR = eye_aspect_ratio(right_eye)

                    avg = (left_EAR + right_EAR) / 2

                    st.toast(f"The EAR of the eyes is: {avg}")

                    if avg < EAR_THRESHOLD:
                        message.error("The eyes are shut!")

                    else:
                        message.success("The eyes are detected!")

                    progress_bar.progress(100, text="Done")

                    image_placeholder.image(image, caption="Uploaded Photo with Faces", use_column_width=True)
                    #
                    # with col1:
                    #     st.image(left_eye_roi)
                    #     st.caption("The left eye")
                    #
                    # with col2:
                    #     st.image(right_eye_roi)
                    #     st.caption("The right eye")

                # else:
                #     st.write("Something went wrong!")

elif selectbox_option == "Take Photo":

    camera_input = st.camera_input("Take a picture of your face")

    progress_bar = st.empty()

    if camera_input is not None:

        progress_bar.progress(0, text="Analyzing the picture.....")

        image_numpy_array = image_to_numpy(camera_input)

        image_gray = cv2.cvtColor(image_numpy_array, cv2.COLOR_BGR2GRAY)

        # image_resolution_checker(image_gray)

        faces = face_detection(image_gray)

        face_found = False
        for face in faces:
            shape = landmark_predictor(image_gray, face)
            shape = face_utils.shape_to_np(shape)

            face_hull = cv2.convexHull(shape)
            cv2.drawContours(image_numpy_array, [face_hull], -1, (0, 255, 0), 2)

        message = st.empty()

        if len(faces) == 1:
            message.success("Success, the face detected! You can perform eye detection!")
            face_found = True
            perform_eye_detection = st.button("Perform eye detection")

        elif len(faces) < 1:
            message.error("No faces are detected, please take another photo!")

        else:
            message.error("Multiple faces are detected, please take another photo!")

        progress_bar.progress(100, text="The picture analyzed!")

        image_placeholder = st.image(image_numpy_array, caption="Uploaded Photo with Faces", use_column_width=True)

        if face_found:
            if perform_eye_detection:
                col1, col2 = st.columns(2)

                for face in faces:

                    progress_bar.progress(0, text="Detecting eyes....")

                    shape = landmark_predictor(image_gray, face)

                    shape = face_utils.shape_to_np(shape)

                    left_eye = shape[L_start: L_end]
                    right_eye = shape[R_start:R_end]

                    left_eye_hull = cv2.convexHull(left_eye)
                    right_eye_hull = cv2.convexHull(right_eye)
                    cv2.drawContours(image_numpy_array, [left_eye_hull], -1, (0, 255, 0), 2)
                    cv2.drawContours(image_numpy_array, [right_eye_hull], -1, (0, 255, 0), 2)

                    left_EAR = eye_aspect_ratio(left_eye)
                    right_EAR = eye_aspect_ratio(right_eye)

                    avg = (left_EAR + right_EAR) / 2

                    st.toast(f"The EAR of the eyes is: {avg}")

                    if avg < EAR_THRESHOLD_WEBCAM:
                        message.error("The eyes are shut!")

                    else:
                        message.success("The eyes are detected!")

                    progress_bar.progress(100, text="Done!")

                    image_placeholder.image(image_numpy_array, caption="Uploaded Photo with Faces", use_column_width=True)