# import os
# import cv2

# # Paths (adjust based on your folder structure)
# frames_word_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Word_Level"  # Folder with word-level frames
# frames_sentence_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Frames_Sentence_Level"  # Folder with sentence-level frames (nested)
# processed_frames_output_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Frames"  # Folder to save processed frames

# # Create the frames output directory if it doesn't exist
# os.makedirs(processed_frames_output_dir, exist_ok=True)

# # Function to process image frames
# def process_image_frames(frame_folder_path, output_folder):
#     frame_count = 0
    
#     # Iterate through image files in the frame folder
#     for frame_file in os.listdir(frame_folder_path):
#         frame_path = os.path.join(frame_folder_path, frame_file)
        
#         # Read and process the image
#         img = cv2.imread(frame_path)
#         if img is None:
#             print(f"Failed to load image: {frame_path}")
#             continue
        
#         # Save the processed image to the output folder
#         output_frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
#         cv2.imwrite(output_frame_path, img)
#         frame_count += 1

#     print(f"Processed {frame_count} frames from {frame_folder_path}")

# # Process word-level frames
# for word_folder in os.listdir(frames_word_dir):
#     word_folder_path = os.path.join(frames_word_dir, word_folder)
    
#     if os.path.isdir(word_folder_path):
#         # Create a directory to store processed frames for each word
#         word_output_folder = os.path.join(processed_frames_output_dir, 'Words', word_folder)
#         os.makedirs(word_output_folder, exist_ok=True)
        
#         # Process the frames in the word folder
#         process_image_frames(word_folder_path, word_output_folder)

# # Process sentence-level frames (nested folders)
# for sentence_folder in os.listdir(frames_sentence_dir):
#     sentence_folder_path = os.path.join(frames_sentence_dir, sentence_folder)
    
#     if os.path.isdir(sentence_folder_path):
#         # Iterate through subfolders inside each sentence folder
#         for subfolder in os.listdir(sentence_folder_path):
#             subfolder_path = os.path.join(sentence_folder_path, subfolder)
            
#             if os.path.isdir(subfolder_path):
#                 # Create a directory to store processed frames for each subfolder (each subfolder has frames)
#                 sentence_output_folder = os.path.join(processed_frames_output_dir, 'Sentences', sentence_folder, subfolder)
#                 os.makedirs(sentence_output_folder, exist_ok=True)
                
#                 # Process the frames in the subfolder
#                 process_image_frames(subfolder_path, sentence_output_folder)

# print("Frame processing for word-level and sentence-level frames complete!")







import os
import cv2
import pandas as pd
from PIL import Image

# Define paths (adjust based on your folder structure)
videos_sentence_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Videos_Sentence_Level"
processed_video_frames_dir = r"C:\Users\Souvik Baidya\Documents\Processed_Video_Frames"  # Your desired output directory
csv_file_path = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\corpus_csv_files\ISL Corpus sign glosses.csv"  # Path to your .csv file

# Load the CSV file (Assuming columns: 'Sentences' and 'SIGN GLOSSES')
labels_df = pd.read_csv(csv_file_path)

# Create the output directory if it doesn't exist
os.makedirs(processed_video_frames_dir, exist_ok=True)

# Function to map sentence folders to their gloss labels from the CSV
def get_sentence_label_from_csv(sentence_name, labels_df):
    matched_row = labels_df[labels_df['Sentence'] == sentence_name]
    if not matched_row.empty:
        return matched_row.iloc[0]['SIGN GLOSSES']  # Assuming 'SIGN GLOSSES' contains labels
    return None

# Function to process videos and extract frames
def process_video_frames(input_dir, output_dir, labels_df, sequence_length=30):
    for sentence_folder in os.listdir(input_dir):
        sentence_folder_path = os.path.join(input_dir, sentence_folder)

        # Get the corresponding label for this sentence from the CSV
        label = get_sentence_label_from_csv(sentence_folder, labels_df)
        
        if label and os.path.isdir(sentence_folder_path):
            for video_file in sorted(os.listdir(sentence_folder_path)):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(sentence_folder_path, video_file)
                    video_name = os.path.splitext(video_file)[0]
                    
                    # Create a directory for this video’s frames
                    video_frames_output_dir = os.path.join(output_dir, label, video_name)
                    os.makedirs(video_frames_output_dir, exist_ok=True)

                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_step = max(1, total_frames // sequence_length)
                    frame_count = 0

                    for i in range(0, total_frames, frame_step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = cap.read()
                        if ret:
                            # Convert frame to PIL image for consistency
                            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            # Save the frame with the label in the filename
                            save_frame(frame_pil, video_frames_output_dir, f"{label}_frame_{frame_count:05d}.jpg")
                            frame_count += 1

                    cap.release()

# Helper function to save frames
def save_frame(frame, output_folder, filename):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, filename)
    frame.save(output_path)

# Process video frames from Videos_Sentence_Level
print("Processing video frames from Videos_Sentence_Level...")
process_video_frames(videos_sentence_dir, processed_video_frames_dir, labels_df)

print("Video frame extraction and processing completed!")
