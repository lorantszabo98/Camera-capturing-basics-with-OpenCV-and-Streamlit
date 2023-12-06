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
- Home.py: Display the real time dataframe from the age and gender detection results and display a chart of the age, gender and the date.

- Age_and_Gender_detection.py: Users can start their webcam to enable face detection and to collect the age and gender data based on the pretrained CAFFE models. The user can enter their name, select what the program should detect and send the data. The program will send the most detected gender or age data.

- Age_and_Gender_detection_Results.py: Here is the real-time user data that users submit with `Age_and_Gender_detection.py`. A button is used to delete the last line, the program also handles the empty dataframe.

- Eye_detection.py: This page can be used for eye detection. It works with two types of input, the user can upload an image from their computer or capture an image of themselves using a webcam. In each case, the program checks that only one face is detected in the image, then uses a button to perform the eye detection and then displays the contours in the image. The program checks whether the eye is closed by calculating the EAR (eye aspect ratio) and indicates this to the user.

- Blink_and_Yawn_detection.py: This screen uses facial landmarks to calculate the EAR (eye aspect ratio) and lip distance to detect blinking and yawning in real time. To do this, it uses dlib's pre-trained facial landmark detector and calculates the distance between the eye points. If this distance decreases significantly from a certain point, it is detected as a blink. The yawn detector works on a similar principle, only there the distance between the mouth points is calculated. If this distance increases gradually (and stays at a higher value for several frames), it is detected as a yawn. It's important to note that both detection algorithms use threshold values for detection, but these threshold values can vary quite a bit depending on different factors. In order to make the setting of these threshold values dynamic, the program spends 5 seconds calibrating to set these values correctly. 
The program counts the total number of blinks and yawns detected, and plots both the EAR value and the lip distance value in real time on an Altair chart.


