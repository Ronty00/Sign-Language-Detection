import os
from sklearn.preprocessing import LabelEncoder
import joblib

# Directories containing the processed images and video frames
processed_image_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Images_Frames"  # Adjust path to your 'Processed_Image_Frames'
processed_video_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Video_Frames"  # Adjust path to your 'Processed_Video_Frames'

# Collect labels from Sentences, Words, and Videos
labels = set()  # Using a set to avoid duplicates

# Extract labels from 'Sentences' and 'Words' folder in 'Processed_Image_Frames'
for folder in ['Sentences', 'Words']:
    folder_path = os.path.join(processed_image_frames_dir, folder)
    if os.path.exists(folder_path):
        for label in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, label)):
                labels.add(label)  # Add the folder name as a label

# Extract labels from 'Processed_Video_Frames' folder
for label in os.listdir(processed_video_frames_dir):
    if os.path.isdir(os.path.join(processed_video_frames_dir, label)):
        labels.add(label)  # Add the folder name as a label

# Convert the set of labels to a sorted list (optional: for consistency)
labels = sorted(list(labels))

# Create and fit the LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)  # Fit the LabelEncoder with the collected labels

# Save the fitted LabelEncoder as a .pkl file
joblib.dump(label_encoder, r"C:/Users/Souvik Baidya/Documents/Software Engineering/Project/ISL Video Dataset/ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/label_encoder.pkl")

print(f"LabelEncoder has been created with {len(labels)} unique labels and saved to label_encoder.pkl")
