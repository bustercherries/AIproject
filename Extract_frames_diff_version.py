import cv2
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1

    cap.release()
    print(f"Frames extracted: {frame_count}")
          
video_path = r"C:\Users\rosie\github repos\AIproject\videos\Dziki.mp4"  
output_folder = "extracted_frames" 
extract_frames(video_path, output_folder)