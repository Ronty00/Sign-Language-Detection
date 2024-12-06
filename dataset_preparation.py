# import numpy as np
# import os
# import cv2
# import pickle

# # Paths for video and image frames
# video_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Video_Frames"
# image_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Images_Frames"  # Contains 'Words' and 'Sentences'
# output_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"  # Specify where you want to save the processed data

# # Ensure the output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Initialize lists for storing data and labels
# X_videos = []  # Stores frames for videos
# X_images = []  # Stores frames for images (word/sentence)
# y_labels = []  # Stores labels for both inputs

# # Function to load frames from a folder
# def load_frames_from_folder(folder_path):
#     frames = []
#     for frame_file in os.listdir(folder_path):
#         frame_path = os.path.join(folder_path, frame_file)
        
#         # Skip non-image files like Thumbs.db
#         if frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#             frame = cv2.imread(frame_path)
#             frame = cv2.resize(frame, (64, 64))  # Resize frames for consistency
#             frames.append(frame)
#     return np.array(frames)

# # 1. Load video frames (Process each video inside each sentence folder)
# for sentence_folder in os.listdir(video_frames_dir):
#     sentence_folder_path = os.path.join(video_frames_dir, sentence_folder)
    
#     for video_folder in os.listdir(sentence_folder_path):
#         video_folder_path = os.path.join(sentence_folder_path, video_folder)
        
#         # Load frames from each video folder
#         video_frames = load_frames_from_folder(video_folder_path)
#         if len(video_frames) > 0:  # Skip empty videos
#             X_videos.append(video_frames)
#             y_labels.append(sentence_folder)  # Label corresponds to sentence folder name

# # 2. Load word-level image frames
# for word_folder in os.listdir(os.path.join(image_frames_dir, 'Words')):
#     word_folder_path = os.path.join(image_frames_dir, 'Words', word_folder)
    
#     # Load frames from each word folder
#     word_frames = load_frames_from_folder(word_folder_path)
#     if len(word_frames) > 0:  # Skip empty folders
#         X_images.append(word_frames)
#         y_labels.append(word_folder)  # Label corresponds to word folder name

# # 3. Load sentence-level image frames (Process each subfolder inside each sentence folder)
# for sentence_folder in os.listdir(os.path.join(image_frames_dir, 'Sentences')):
#     sentence_folder_path = os.path.join(image_frames_dir, 'Sentences', sentence_folder)
    
#     for subfolder in os.listdir(sentence_folder_path):
#         subfolder_path = os.path.join(sentence_folder_path, subfolder)
        
#         # Load frames from each subfolder inside the sentence folder
#         sentence_frames = load_frames_from_folder(subfolder_path)
#         if len(sentence_frames) > 0:  # Skip empty subfolders
#             X_images.append(sentence_frames)
#             y_labels.append(sentence_folder)  # Label corresponds to the sentence folder name

# # 4. Padding to Handle Inhomogeneous Shape Issue
# # Find the maximum number of frames in the video and image data
# max_video_frames = max([len(video) for video in X_videos]) if X_videos else 0
# max_image_frames = max([len(img) for img in X_images]) if X_images else 0

# # Pad video frames to have uniform length
# X_videos_padded = []
# for video in X_videos:
#     if len(video) < max_video_frames:
#         padding = np.zeros((max_video_frames - len(video), 64, 64, 3), dtype=np.uint8)  # Adjust based on frame shape
#         video_padded = np.concatenate((video, padding), axis=0)
#     else:
#         video_padded = video
#     X_videos_padded.append(video_padded)

# # Pad image frames to have uniform length
# X_images_padded = []
# for img in X_images:
#     if len(img) < max_image_frames:
#         padding = np.zeros((max_image_frames - len(img), 64, 64, 3), dtype=np.uint8)
#         img_padded = np.concatenate((img, padding), axis=0)
#     else:
#         img_padded = img
#     X_images_padded.append(img_padded)

# # Convert padded data to numpy arrays
# X_videos_padded = np.array(X_videos_padded)
# X_images_padded = np.array(X_images_padded)
# y_labels = np.array(y_labels)

# # Save the padded datasets
# np.save(os.path.join(output_dir, 'X_videos_padded.npy'), X_videos_padded)
# np.save(os.path.join(output_dir, 'X_images_padded.npy'), X_images_padded)
# np.save(os.path.join(output_dir, 'y_labels.npy'), y_labels)

# print(f"Padded data saved in {output_dir} as 'X_videos_padded.npy', 'X_images_padded.npy', and 'y_labels.npy'")
import os
import numpy as np
from PIL import Image
import cv2

# Function to create output directories if they don't exist
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Paths (replace with your actual paths)
processed_image_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Images_Frames"
processed_video_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Video_Frames"
output_dataset_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus"  # Desired output directory

# Create output directory if it doesn't exist
create_directory(output_dataset_dir)

# Function to process sentence images and save directly to .npy
def process_sentence_images(sentence_dir):
    all_images = []
    all_labels = []
    
    for sentence_folder in os.listdir(sentence_dir):
        sentence_folder_path = os.path.join(sentence_dir, sentence_folder)
        if os.path.isdir(sentence_folder_path):
            for example_folder in os.listdir(sentence_folder_path):
                example_folder_path = os.path.join(sentence_folder_path, example_folder)
                if os.path.isdir(example_folder_path):
                    for image_file in os.listdir(example_folder_path):
                        image_path = os.path.join(example_folder_path, image_file)
                        if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            image = Image.open(image_path)
                            # Resize the image to reduce memory usage
                            image = image.resize((64, 64), Image.LANCZOS)
                            all_images.append(np.array(image))  # Store image as numpy array
                            all_labels.append(sentence_folder)  # Store the label

    return np.array(all_images, dtype=object), np.array(all_labels)

# Function to process word images and save directly to .npy
def process_word_images(word_dir):
    all_images = []
    all_labels = []
    
    for word_folder in os.listdir(word_dir):
        word_folder_path = os.path.join(word_dir, word_folder)
        if os.path.isdir(word_folder_path):
            for image_file in os.listdir(word_folder_path):
                image_path = os.path.join(word_folder_path, image_file)
                if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    image = Image.open(image_path)
                    # Resize the image to reduce memory usage
                    image = image.resize((64, 64), Image.LANCZOS)
                    all_images.append(np.array(image))  # Store image as numpy array
                    all_labels.append(word_folder)  # Store the label

    return np.array(all_images, dtype=object), np.array(all_labels)

# Function to process video frames and save directly to .npy
def process_video_frames(video_dir):
    all_frames = []
    all_labels = []
    
    for sentence_folder in os.listdir(video_dir):
        sentence_folder_path = os.path.join(video_dir, sentence_folder)
        if os.path.isdir(sentence_folder_path):
            for video_folder in os.listdir(sentence_folder_path):
                video_folder_path = os.path.join(sentence_folder_path, video_folder)
                if os.path.isdir(video_folder_path):
                    for frame_file in os.listdir(video_folder_path):
                        frame_path = os.path.join(video_folder_path, frame_file)
                        if frame_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            frame = cv2.imread(frame_path)
                            frame = cv2.resize(frame, (64, 64))  # Resize the frame
                            all_frames.append(frame)  # Store frame as numpy array
                            all_labels.append(sentence_folder)  # Store the label

    return np.array(all_frames, dtype=object), np.array(all_labels)

# Process all the data and save them as .npy files
print("Processing sentence images...")
sentence_images_np, sentence_labels_np = process_sentence_images(os.path.join(processed_image_frames_dir, "Sentences"))

print("Processing word images...")
word_images_np, word_labels_np = process_word_images(os.path.join(processed_image_frames_dir, "Words"))

print("Processing video frames...")
video_frames_np, video_labels_np = process_video_frames(processed_video_frames_dir)

# Save the processed data to .npy files
np.save(os.path.join(output_dataset_dir, 'sentence_images.npy'), sentence_images_np)
np.save(os.path.join(output_dataset_dir, 'word_images.npy'), word_images_np)
np.save(os.path.join(output_dataset_dir, 'video_frames.npy'), video_frames_np)

np.save(os.path.join(output_dataset_dir, 'sentence_labels.npy'), sentence_labels_np)
np.save(os.path.join(output_dataset_dir, 'word_labels.npy'), word_labels_np)
np.save(os.path.join(output_dataset_dir, 'video_labels.npy'), video_labels_np)

print("Labeled dataset saved as .npy files in", output_dataset_dir)
