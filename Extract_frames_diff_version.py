import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        # Check if the frame was read successfully
        if not ret:
            break
        
        # Write the frame to a file
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    print(f"Frames extracted: {frame_count}")

# Example usage
video_path = r"C:\Users\rosie\github repos\AIproject\videos\Daniele_2023.mp4"  # Path to the input video file
output_folder = "extracted_frames"  # Output folder to save frames
extract_frames(video_path, output_folder)