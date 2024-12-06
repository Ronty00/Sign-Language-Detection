import os
import cv2

# Paths (adjust based on your folder structure)
video_sentence_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level"  # Folder with sentence-level video folders
frames_output_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Video_Frames"  # Folder to save extracted frames

# Create the frames output directory if it doesn't exist
os.makedirs(frames_output_dir, exist_ok=True)

# Parameters
num_frames_to_extract = 30  # Number of frames to extract per video
valid_video_extensions = ['.mp4', '.avi', '.mkv', '.mov']  # Valid video extensions

# Function to extract frames from a video
def extract_frames_from_video(video_path, output_folder, num_frames_to_extract):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"No frames found in: {video_path}")
        return None

    step = max(1, total_frames // num_frames_to_extract)
    frame_count = 0

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        if frame_count >= num_frames_to_extract:
            break

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

# Iterate through each sentence folder
for sentence_folder in os.listdir(video_sentence_dir):
    sentence_folder_path = os.path.join(video_sentence_dir, sentence_folder)
    
    if os.path.isdir(sentence_folder_path):
        # Create a directory to store frames for each sentence
        sentence_output_folder = os.path.join(frames_output_dir, sentence_folder)
        os.makedirs(sentence_output_folder, exist_ok=True)
        
        # Iterate through each video in the sentence folder
        for video_name in os.listdir(sentence_folder_path):
            video_path = os.path.join(sentence_folder_path, video_name)
            
            # Check if the file is a valid video
            if any(video_name.lower().endswith(ext) for ext in valid_video_extensions):
                # Create a subfolder for each video inside the sentence folder
                video_output_folder = os.path.join(sentence_output_folder, video_name.split('.')[0])
                os.makedirs(video_output_folder, exist_ok=True)

                # Extract frames from the video
                extract_frames_from_video(video_path, video_output_folder, num_frames_to_extract)
            else:
                print(f"Skipping non-video file: {video_name}")

print("Video frame extraction complete!")
