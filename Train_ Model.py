# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# import numpy as np
# import joblib
# from PIL import Image

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Define the combined CNN + LSTM model with fine-tuning
# class CNNLSTMModel(nn.Module):
#     def __init__(self, num_classes, hidden_dim, num_layers):
#         super(CNNLSTMModel, self).__init__()
#         # Pre-trained ResNet18 as the CNN for feature extraction
#         self.cnn = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
#         self.cnn.fc = nn.Identity()  # Remove the final classification layer

#         # Fine-tune only the last ResNet layers
#         for param in self.cnn.parameters():
#             param.requires_grad = False
#         for param in self.cnn.layer4.parameters():
#             param.requires_grad = True  # Fine-tuning the last layer

#         self.flattened_size = 2048  # ResNet18 output size
        
#         # LSTM layers
#         self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.5, batch_first=True)
        
#         # Fully connected layer
#         self.fc = nn.Linear(hidden_dim, num_classes)
    
#     def forward(self, x):
#         batch_size, sequence_length, _, _, _ = x.size()
#         cnn_features = []

#         # Process each frame through the ResNet
#         for i in range(sequence_length):
#             frame = x[:, i, :, :, :]  # Shape: (batch_size, C, H, W)
#             frame = self.cnn(frame)
#             cnn_features.append(frame)

#         # Stack CNN features along time dimension
#         cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (batch_size, seq_len, flattened_size)

#         # LSTM sequence processing
#         lstm_out, _ = self.lstm(cnn_features)
#         last_lstm_output = lstm_out[:, -1, :]  # Take the last output of LSTM

#         # Pass the LSTM output through the fully connected layer
#         output = self.fc(last_lstm_output)
        
#         return output

# # Custom dataset for loading sequences of frames
# class SignLanguageDataset(Dataset):
#     def __init__(self, frame_paths, labels, transform=None, sequence_length=30):
#         self.frame_paths = frame_paths  # List of lists of frame paths (images)
#         self.labels = labels  # Corresponding labels for each sequence
#         self.transform = transform
#         self.sequence_length = sequence_length
    
#     def __len__(self):
#         return len(self.labels)
    
#     def __getitem__(self, idx):
#         frames = []
#         frame_paths = self.frame_paths[idx]
#         frame_paths_length = len(frame_paths)

#         if frame_paths_length == 0:
#             return None, None  # Skip empty sequences

#         # Load and transform each frame in the sequence
#         for i in range(self.sequence_length):
#             if i < frame_paths_length:
#                 frame_path = frame_paths[i]
#                 if os.path.exists(frame_path):
#                     image = Image.open(frame_path)
#                     if self.transform:
#                         image = self.transform(image)
#                     frames.append(image)
#             else:
#                 # Pad with the last frame if sequence is too short
#                 frames.append(frames[-1])
        
#         frames = torch.stack(frames)  # Shape: (sequence_length, C, H, W)
#         label = self.labels[idx]
#         return frames, label

# # Function to load sequences of frames from Processed_Video_Frames and Processed_Images_Frames
# def load_combined_data(processed_video_dir, processed_image_dir, sequence_length=30):
#     frame_sequences = []  # List of lists of frame paths for videos and images
#     labels = []  # Corresponding labels for each sequence
    
#     # Process video directories
#     for label_folder in os.listdir(processed_video_dir):
#         label_folder_path = os.path.join(processed_video_dir, label_folder)
        
#         if os.path.isdir(label_folder_path):
#             for video_folder in os.listdir(label_folder_path):
#                 video_folder_path = os.path.join(label_folder_path, video_folder)
                
#                 if os.path.isdir(video_folder_path):
#                     frame_paths = []
#                     for frame_file in sorted(os.listdir(video_folder_path)):
#                         frame_path = os.path.join(video_folder_path, frame_file)
#                         if frame_file.endswith(('.jpg', '.jpeg', '.png')) and os.path.exists(frame_path):
#                             frame_paths.append(frame_path)
                    
#                     if len(frame_paths) > 0:
#                         frame_sequences.append(frame_paths[:sequence_length])
#                         labels.append(label_folder)
#                     else:
#                         print(f"Skipping {video_folder_path}, no frames found.")

#     return frame_sequences, labels

# # Data Augmentation
# transform = transforms.Compose([
#     transforms.Resize((80, 60)),  # Resize to match model input
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(20),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
#     transforms.RandomAffine(20),  # Add random affine transformations
#     transforms.ToTensor(),
# ])

# # Load dataset
# processed_video_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Video_Frames"
# processed_images_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Images_Frames"
# frame_paths, labels = load_combined_data(processed_video_frames_dir, processed_images_frames_dir, sequence_length=30)

# # Split dataset into training and validation sets (80% train, 20% validation)
# train_frame_paths, val_frame_paths, train_labels, val_labels = train_test_split(
#     frame_paths, labels, test_size=0.2, random_state=42)

# # Convert labels to numerical values
# label_encoder = LabelEncoder()
# label_encoder.fit(labels)
# train_labels = label_encoder.transform(train_labels)
# val_labels = label_encoder.transform(val_labels)

# # Save the label encoder
# joblib.dump(label_encoder, r'C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\label_encoder.pkl')

# # Create datasets and dataloaders
# train_dataset = SignLanguageDataset(train_frame_paths, train_labels, transform=transform, sequence_length=30)
# val_dataset = SignLanguageDataset(val_frame_paths, val_labels, transform=transform, sequence_length=30)

# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# # Initialize the model
# model = CNNLSTMModel(num_classes=len(label_encoder.classes_), hidden_dim=1024, num_layers=6).to(device)

# # Loss and optimizer with weight decay
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# # Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# # Training the model with validation accuracy tracking and gradient clipping
# def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=2000):
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for frames, labels in train_loader:
#             if frames is None:  # Skip empty sequences
#                 continue
#             frames, labels = frames.to(device), labels.to(device).long()
            
#             # Forward pass
#             outputs = model(frames)
#             loss = criterion(outputs, labels)
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()

#             # Gradient clipping
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#             optimizer.step()
            
#             # Track training accuracy
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             running_loss += loss.item()
        
#         train_accuracy = 100 * correct / total
#         train_loss = running_loss / total
        
#         # Validation phase
#         model.eval()
#         val_correct = 0
#         val_total = 0
#         val_loss = 0
#         with torch.no_grad():
#             for val_frames, val_labels in val_loader:
#                 if val_frames is None:  # Skip empty sequences
#                     continue
#                 val_frames, val_labels = val_frames.to(device), val_labels.to(device).long()

#                 val_outputs = model(val_frames)
#                 _, val_predicted = torch.max(val_outputs, 1)
                
#                 val_loss += criterion(val_outputs, val_labels).item()
#                 val_total += val_labels.size(0)
#                 val_correct += (val_predicted == val_labels).sum().item()

#         val_accuracy = 100 * val_correct / val_total
#         val_loss /= val_total

#         # Print epoch stats
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
#               f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}")
        
#         # Adjust learning rate based on validation loss
#         scheduler.step(val_loss)

# # Train the model
# train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=2000)

# # Save the trained model
# torch.save(model.state_dict(), r'C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\sign_language_cnn_lstm_model_3.pth')



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from PIL import Image
import cv2
import mediapipe as mp

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize MediaPipe for hand, face, and body detection
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
mp_face_detection = mp.solutions.face_detection.FaceDetection()
mp_pose = mp.solutions.pose.Pose(static_image_mode=False)

# Function to detect and crop hands, face, and body using MediaPipe
def detect_and_crop_key_regions(frame):
    frame_rgb = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
    
    # Detect hands
    hand_results = mp_hands.process(frame_rgb)
    hands = []
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hands.append(hand_landmarks)

    # Detect face
    face_results = mp_face_detection.process(frame_rgb)
    face = None
    if face_results.detections:
        face = face_results.detections[0]
    
    # Detect pose (body)
    pose_results = mp_pose.process(frame_rgb)
    body = None
    if pose_results.pose_landmarks:
        body = pose_results.pose_landmarks

    # Return detected regions (you can crop these regions here as needed)
    return hands, face, body

# CNN + LSTM model definition
class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_layers):
        super(CNNLSTMModel, self).__init__()
        # Pre-trained ResNet152 as the CNN for feature extraction
        self.cnn = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Remove the final classification layer

        # Fine-tune only the last ResNet layers
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.cnn.layer4.parameters():
            param.requires_grad = True  # Fine-tuning the last layer

        self.flattened_size = 2048  # ResNet152 output size
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.5, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        batch_size, sequence_length, _, _, _ = x.size()
        cnn_features = []

        # Process each frame through the ResNet
        for i in range(sequence_length):
            frame = x[:, i, :, :, :]  # Shape: (batch_size, C, H, W)
            frame = self.cnn(frame)
            cnn_features.append(frame)

        # Stack CNN features along time dimension
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (batch_size, seq_len, flattened_size)

        # LSTM sequence processing
        lstm_out, _ = self.lstm(cnn_features)
        last_lstm_output = lstm_out[:, -1, :]  # Take the last output of LSTM

        # Pass the LSTM output through the fully connected layer
        output = self.fc(last_lstm_output)
        
        return output

# Custom dataset for loading sequences of frames
class SignLanguageDataset(Dataset):
    def __init__(self, frame_paths, labels, transform=None, sequence_length=30):
        self.frame_paths = frame_paths  # List of lists of frame paths (images)
        self.labels = labels  # Corresponding labels for each sequence
        self.transform = transform
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        frames = []
        frame_paths = self.frame_paths[idx]
        frame_paths_length = len(frame_paths)

        if frame_paths_length == 0:
            return None, None  # Skip empty sequences

        # Load and transform each frame in the sequence
        for i in range(self.sequence_length):
            if i < frame_paths_length:
                frame_path = frame_paths[i]
                if os.path.exists(frame_path):
                    image = Image.open(frame_path)
                    
                    # Detect hands, face, and body and crop relevant regions
                    hands, face, body = detect_and_crop_key_regions(image)

                    if self.transform:
                        image = self.transform(image)
                    frames.append(image)
            else:
                # Pad with the last frame if sequence is too short
                frames.append(frames[-1])
        
        frames = torch.stack(frames)  # Shape: (sequence_length, C, H, W)
        label = self.labels[idx]
        return frames, label

# Function to load sequences of frames from Processed_Video_Frames and Processed_Images_Frames
def load_combined_data(processed_video_dir, processed_image_dir, sequence_length=30):
    frame_sequences = []  # List of lists of frame paths for videos and images
    labels = []  # Corresponding labels for each sequence
    
    # Process video directories
    for label_folder in os.listdir(processed_video_dir):
        label_folder_path = os.path.join(processed_video_dir, label_folder)
        
        if os.path.isdir(label_folder_path):
            for video_folder in os.listdir(label_folder_path):
                video_folder_path = os.path.join(label_folder_path, video_folder)
                
                if os.path.isdir(video_folder_path):
                    frame_paths = []
                    for frame_file in sorted(os.listdir(video_folder_path)):
                        frame_path = os.path.join(video_folder_path, frame_file)
                        if frame_file.endswith(('.jpg', '.jpeg', '.png')) and os.path.exists(frame_path):
                            frame_paths.append(frame_path)
                    
                    if len(frame_paths) > 0:
                        frame_sequences.append(frame_paths[:sequence_length])
                        labels.append(label_folder)
                    else:
                        print(f"Skipping {video_folder_path}, no frames found.")

    return frame_sequences, labels

# Data Augmentation
transform = transforms.Compose([
    transforms.Resize((320, 240)),  # Resize to match model input
    transforms.ToTensor(),
])

# Load dataset
processed_video_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Video_Frames"
processed_images_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Images_Frames"
frame_paths, labels = load_combined_data(processed_video_frames_dir, processed_images_frames_dir, sequence_length=30)

# Split dataset into training and validation sets (80% train, 20% validation)
train_frame_paths, val_frame_paths, train_labels, val_labels = train_test_split(
    frame_paths, labels, test_size=0.2, random_state=42)

# Convert labels to numerical values
label_encoder = LabelEncoder()
label_encoder.fit(labels)
train_labels = label_encoder.transform(train_labels)
val_labels = label_encoder.transform(val_labels)

# Save the label encoder
joblib.dump(label_encoder, r'C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\label_encoder.pkl')

# Create datasets and dataloaders
train_dataset = SignLanguageDataset(train_frame_paths, train_labels, transform=transform, sequence_length=30)
val_dataset = SignLanguageDataset(val_frame_paths, val_labels, transform=transform, sequence_length=30)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize the model
model = CNNLSTMModel(num_classes=len(label_encoder.classes_), hidden_dim=1024, num_layers=6).to(device)

# Loss and optimizer with weight decay
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# Training the model with validation accuracy tracking and gradient clipping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=2000):
    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for frames, labels in train_loader:
            if frames is None:  # Skip empty sequences
                continue
            frames, labels = frames.to(device), labels.to(device).long()
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            # Track training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for val_frames, val_labels in val_loader:
                if val_frames is None:  # Skip empty sequences
                    continue
                val_frames, val_labels = val_frames.to(device), val_labels.to(device).long()

                val_outputs = model(val_frames)
                _, val_predicted = torch.max(val_outputs, 1)
                
                val_loss += criterion(val_outputs, val_labels).item()
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= val_total

        # Print epoch stats
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
              f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}")
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Save the model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), r'C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\best_sign_language_cnn_lstm_model.pth')

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)

# Save the final model
torch.save(model.state_dict(), r'C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\final_sign_language_cnn_lstm_model.pth')
