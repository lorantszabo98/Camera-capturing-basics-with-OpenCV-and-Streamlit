# Camera-capturing-basics-with-OpenCV-and-Streamlit

This Streamlit app is designed for capturing webcam feed, detecting faces, and collecting gender and age information from users. It combines face detection with gender and age classification using pre-trained models.

# Features
- Webcam Capture: Users can start their webcam to enable face detection and information collection.

- User Information Collection: Users can enter their name and submit it. The app then captures their gender and age information based on face detection.

- Face Detection: The app uses a pre-trained face detection model to identify faces in the webcam feed.

- Gender and Age Classification: Gender and age are classified using pre-trained models, and the most common detection results are stored.

- Blink Detection: Blink detection using dlib facial landmarks and EAR(Eye asprct ratio) to determine a blink.

# Instructions
Installation:

- Ensure you have Python installed.
 
- Install required libraries, like Streamlit, Opencv, dlib...

- Run the App:
  - Execute streamlit run your_app_filename.py in the terminal.
  - Access the app in your browser at http://localhost:8501.