import cv2
import os

def extract_frames(video_path, output_folder, file_name):
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
        
        # Only save every 10th frame
        if frame_count % 10 == 0:
            frame_path = os.path.join(output_folder, f"Jelen_{file_name}_frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        
        frame_count += 1
    cap.release()
    print(f"Frames extracted: {frame_count}")

def extract_frames_list(folder_path, output_folder):
    video_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    for idx, video_path in enumerate(video_paths):
        name = f"Video_{idx+1}"
        extract_frames(video_path, output_folder, name)

folder_path = 'Filmy_Z_Youtube\jelenie_test'
output_folder = "Klatki_Z_Youtube\jelenie_test"
extract_frames_list(folder_path, output_folder)