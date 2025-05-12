import cv2
import numpy as np
from deepface import DeepFace
import time

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting facial expression detection...")
    print("Press 'q' to quit")
    
    # Set frame dimensions (can be adjusted)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Dictionary to store emotion colors
    emotion_colors = {
        "happy": (0, 255, 0),      # Green
        "sad": (0, 0, 255),        # Red
        "angry": (0, 0, 255),      # Red
        "fear": (255, 0, 255),     # Purple
        "disgust": (0, 128, 128),  # Teal
        "surprise": (255, 165, 0), # Orange
        "neutral": (255, 255, 0)   # Yellow
    }
    
    # Limit analysis frequency to reduce CPU usage
    last_detection_time = time.time()
    detection_interval = 0.5  # seconds
    
    # Store last detected emotion to display when no face is detected
    last_emotion = "happy"
    
    # For tracking emotion confidence
    emotion_confidence = 0.0
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Create a copy of the frame to display results
        display_frame = frame.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        current_time = time.time()
        
        # If face(s) detected and enough time has passed since last analysis
        if len(faces) > 0 and (current_time - last_detection_time) >= detection_interval:
            try:
                # Analyze using DeepFace
                result = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'],
                    enforce_detection=False,
                    silent=True
                )
                
                # Get dominant emotion
                emotions = result[0]['emotion']
                highest_score = -1
                dominant_emotion = None
                
                # Get the dominant emotion directly from DeepFace
                for emotion, score in emotions.items():
                    if score > highest_score:
                        highest_score = score
                        dominant_emotion = emotion
                
                # Use the full range of emotions
                last_emotion = dominant_emotion  # This will be one of: angry, disgust, fear, happy, sad, surprise, neutral
                emotion_confidence = highest_score
                    
                last_detection_time = current_time
                
            except Exception as e:
                print(f"Error in emotion detection: {str(e)}")
        
        # Draw rectangles and labels for all detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Display emotion label with appropriate color for each emotion
            emotion_colors = {
                "happy": (0, 255, 0),      # Green
                "sad": (0, 0, 255),        # Red
                "angry": (0, 0, 255),      # Red
                "fear": (255, 0, 255),     # Purple
                "disgust": (0, 128, 128),  # Teal
                "surprise": (255, 165, 0), # Orange
                "neutral": (255, 255, 0)   # Yellow
            }
            
            # Get the color for the detected emotion (default to white if not in our color map)
            emotion_color = emotion_colors.get(last_emotion, (255, 255, 255))
            
            # Display the emotion text with confidence
            cv2.putText(display_frame, f"{last_emotion} ({emotion_confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # Display status and instructions
        cv2.putText(display_frame, f"Detected emotion: {last_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Confidence: {emotion_confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add emotion legend
        legend_y = 90
        for i, (emotion, color) in enumerate(emotion_colors.items()):
            cv2.putText(display_frame, emotion, (10, legend_y + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        cv2.putText(display_frame, "Press 'q' to exit", (10, legend_y + len(emotion_colors)*25 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Facial Expression Detection', display_frame)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Application terminated")

if __name__ == "__main__":
    main()