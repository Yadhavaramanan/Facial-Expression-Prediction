# Face Expression Prediction

This Python project uses your webcam to detect your face and predict your current facial expression (like happy, sad, angry, etc.) in real-time using DeepFace and OpenCV.

##  Features

- Real-time facial emotion detection
- Uses DeepFace for emotion analysis
- Face detection using OpenCV
- Shows emotion label with confidence score
- Colored emotion labels for better visualization
- Press 'q' to quit the application

##  Requirements

Make sure you have Python installed. Then install the required packages:

```bash
pip install opencv-python deepface numpy
```

You may also need the following (for some environments):

```bash
pip install tensorflow keras
```

##  How to Run

1. Save the code in a file named `face_expression.py`.
2. Open a terminal or command prompt.
3. Run the script:

```bash
python face_expression.py
```

4. A window will open showing your webcam feed.
5. Your facial expression will be detected and shown with confidence.
6. Press `q` to exit the application.

##  Emotion Colors

Each emotion is shown in a different color:

- Happy: Green
- Sad/Angry: Red
- Fear: Purple
- Disgust: Teal
- Surprise: Orange
- Neutral: Yellow

##  Screenshot 

!(First_Screenshot) [img1.png]
!(Second_Screenshot) [img2.png]
!(Three_Screenshot) [img3.png]

##  License

This project is for educational purposes only.
