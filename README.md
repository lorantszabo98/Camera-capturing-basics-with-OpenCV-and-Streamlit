# Camera-capturing-basics-with-OpenCV-and-Streamlit

This Streamlit application contains demos of various image processing tasks.

It can be used for webcam capture, face detection, gender and age classification, eye detection, blink and yawn detection. In all cases the program uses pre-trained models.

# Requirements
- Python
- Streamlit
- OpenCV
- dlib
- imutils
- numpy
- pandas

# Instructions
Running:
  - Execute `streamlit run your_app_filename\Home.py` in the terminal.
  - Access the app in your browser at http://localhost:8501.


# Features
- Webcam Capture: Users can start their webcam to enable face detection and information collection.

- User Information Collection: Users can enter their name and submit it. The app then captures their gender and age information based on face detection.

- Face Detection: The app uses a pre-trained face detection model to identify faces in the webcam feed.

- Gender and Age Classification: Gender and age are classified using pre-trained models, and the most common detection results are stored.

- Blink Detection: Blink detection using dlib facial landmarks and EAR(Eye aspect ratio) to determine a blink.

