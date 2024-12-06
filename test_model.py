# import numpy as np

# # Load the .npy files
# X_videos = np.load(r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\X_videos_padded.npy")
# X_images = np.load(r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\X_images_padded.npy")
# y_labels = np.load(r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\y_labels.npy")

# # Check shapes of the arrays
# print(f'Shape of X_videos: {X_videos.shape}')  # Expected: (num_samples, time_steps, height, width, channels)
# print(f'Shape of X_images: {X_images.shape}')  # Expected: (num_samples * num_frames, height, width, channels)
# print(f'Shape of y_labels: {y_labels.shape}')  # Expected: (num_samples,)

# # Check the first few labels to see if they make sense
# print('First 10 labels:', y_labels[:10])

# # Check if the number of samples matches between X_videos and y_labels
# if len(X_videos) == len(y_labels):
#     print('Number of samples in X_videos and y_labels match.')
# else:
#     print(f'Mismatch! X_videos has {len(X_videos)} samples, but y_labels has {len(y_labels)}.')

# # Similarly, check X_images and y_labels
# if len(X_images) // len(X_videos) == X_videos.shape[1]:  # The number of frames per video
#     print('Number of images in X_images matches the number of frames in X_videos.')
# else:
#     print(f'Mismatch! X_images has {len(X_images)} images, but expected {len(X_videos) * X_videos.shape[1]}.')

# # Sanity check: View the first sample from X_videos and X_images
# import matplotlib.pyplot as plt

# # First video
# first_video = X_videos[0]  # First video sample
# print('Shape of first video:', first_video.shape)

# # Plot first frame of the first video
# plt.imshow(first_video[0])  # First frame (time_steps = 0)
# plt.title('First Frame of First Video')
# plt.show()

# # Plot first image from X_images
# plt.imshow(X_images[0])  # First image from the X_images array
# plt.title('First Image from X_images')
# plt.show()






# import os
# from PIL import Image
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# from torchvision import transforms
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # Custom dataset for loading images from processed frames
# class SignLanguageDataset(Dataset):
#     def __init__(self, image_frames_dir, video_frames_dir, transform=None):
#         self.image_frames_dir = image_frames_dir
#         self.video_frames_dir = video_frames_dir
#         self.transform = transform
#         self.data = []
#         self.labels = []
#         self.label_encoder = LabelEncoder()

#         # Prepare data and labels for image frames (Sentences and Words)
#         self._load_image_frames(os.path.join(image_frames_dir, 'Sentences'), is_sentence=True)
#         self._load_image_frames(os.path.join(image_frames_dir, 'Words'), is_sentence=False)

#         # Prepare data and labels for video frames
#         self._load_video_frames(video_frames_dir)

#         # Convert labels to numerical format using LabelEncoder
#         self.labels = self.label_encoder.fit_transform(self.labels)

#     def _load_image_frames(self, folder, is_sentence):
#         for label_folder in os.listdir(folder):
#             label_folder_path = os.path.join(folder, label_folder)
#             if os.path.isdir(label_folder_path):
#                 for subfolder in os.listdir(label_folder_path):
#                     subfolder_path = os.path.join(label_folder_path, subfolder)
#                     if os.path.isdir(subfolder_path):
#                         for image_file in os.listdir(subfolder_path):
#                             image_path = os.path.join(subfolder_path, image_file)
#                             if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
#                                 self.data.append(image_path)
#                                 self.labels.append(label_folder)  # Label is the name of the folder (sentence or word)

#     def _load_video_frames(self, folder):
#         for label_folder in os.listdir(folder):
#             label_folder_path = os.path.join(folder, label_folder)
#             if os.path.isdir(label_folder_path):
#                 for subfolder in os.listdir(label_folder_path):
#                     subfolder_path = os.path.join(label_folder_path, subfolder)
#                     if os.path.isdir(subfolder_path):
#                         for frame_file in os.listdir(subfolder_path):
#                             frame_path = os.path.join(subfolder_path, frame_file)
#                             if frame_file.lower().endswith(('.jpg', '.png', '.jpeg')):
#                                 self.data.append(frame_path)
#                                 self.labels.append(label_folder)  # Label is the name of the folder (sentence)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image_path = self.data[idx]
#         label = self.labels[idx]

#         # Load the image
#         image = Image.open(image_path)
#         if self.transform:
#             image = self.transform(image)

#         return image, torch.tensor(label, dtype=torch.long)  # Convert label to tensor for classification


# def calculate_accuracy(loader, model, device):
#     model.eval()  # Set the model to evaluation mode
#     correct = 0
#     total = 0
#     with torch.no_grad():  # Disable gradient computation
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return 100 * correct / total

# if __name__ == '__main__':
#     # Define transformations (resize, normalize, etc.)
#     transform = transforms.Compose([
#         transforms.Resize((80, 60)),  # Resize to 80x60, or remove this line to keep original size
#         transforms.ToTensor(),  # Convert image to tensor
#     ])

#     # Paths to your processed folders
#     image_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Images_Frames"
#     video_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Video_Frames"

#     # Create the dataset
#     dataset = SignLanguageDataset(image_frames_dir, video_frames_dir, transform=transform)

#     # Split dataset into training and validation sets
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#     # Create DataLoader for batching and shuffling
#     batch_size = 32
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     # Define a simple CNN model
#     class SignLanguageCNN(nn.Module):
#         def __init__(self):
#             super(SignLanguageCNN, self).__init__()
#             self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#             self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#             self.pool = nn.MaxPool2d(2, 2)
#             self.flattened_size = 64 * 20 * 15  # Adjust based on input size after conv layers
#             self.fc1 = nn.Linear(self.flattened_size, 128)
#             self.fc2 = nn.Linear(128, len(set(dataset.labels)))  # Number of classes

#         def forward(self, x):
#             x = self.pool(nn.ReLU()(self.conv1(x)))
#             x = self.pool(nn.ReLU()(self.conv2(x)))
#             x = x.view(x.size(0), -1)  # Flatten the tensor
#             x = nn.ReLU()(self.fc1(x))
#             x = self.fc2(x)
#             return x

#     # Initialize the model, loss function, and optimizer
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SignLanguageCNN().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Training loop with validation accuracy
#     num_epochs = 50
#     for epoch in range(num_epochs):
#         model.train()  # Set the model to training mode
#         running_loss = 0.0

#         # Training phase
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         # Calculate training and validation accuracy after each epoch
#         train_accuracy = calculate_accuracy(train_loader, model, device)
#         val_accuracy = calculate_accuracy(val_loader, model, device)

#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, "
#               f"Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

#     # Save the trained model
#     model_save_path = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\sign_language_model_3.pth"
#     torch.save(model.state_dict(), model_save_path)
#     print(f"Model saved at {model_save_path}")








import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")
# Define the combined CNN + LSTM model for sequence prediction
class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes, hidden_dim, num_layers):
        super(CNNLSTMModel, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.flattened_size = 64 * 20 * 15  # Adjust based on input size after conv layers
        self.dropout = nn.Dropout(0.5)
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=self.flattened_size, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.3, batch_first=True)
        
        # Fully connected layer to classify sequences
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        batch_size, sequence_length, _, _, _ = x.size()  # Skip unused dimensions
        
        # Process each frame in the sequence through the CNN
        cnn_features = []
        for i in range(sequence_length):
            frame = x[:, i, :, :, :]  # Shape (batch_size, C, H, W)
            frame = self.pool(nn.ReLU()(self.bn1(self.conv1(frame))))
            frame = self.pool(nn.ReLU()(self.bn2(self.conv2(frame))))
            frame = frame.view(frame.size(0), -1)  # Flatten the frame
            cnn_features.append(frame)
        
        # Stack CNN features along the time dimension
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape (batch_size, seq_len, flattened_size)
        
        # Pass the CNN features through the LSTM
        lstm_out, _ = self.lstm(cnn_features)
        
        # Take the last output of the LSTM (which corresponds to the whole sequence)
        last_lstm_output = lstm_out[:, -1, :]  # Shape (batch_size, hidden_dim)
        
        # Pass the last LSTM output through the fully connected layer to classify the sequence
        output = self.fc(last_lstm_output)  # Shape (batch_size, num_classes)
        
        return output

# Custom dataset for loading sequences of frames
class SignLanguageDataset(Dataset):
    def __init__(self, video_frame_paths, labels, transform=None, sequence_length=30):
        self.video_frame_paths = video_frame_paths  # List of lists of frame paths (images)
        self.labels = labels  # Corresponding labels for each sequence
        self.transform = transform
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        frames = []
        video_frame_paths = self.video_frame_paths[idx]
        
        # Ensure enough frames are available for each sequence
        frame_paths_length = len(video_frame_paths)
        if frame_paths_length == 0:
            print(f"Skipping index {idx}, no frames found.")
            return None, None  # Skip empty sequences
        
        # Load and transform each frame in the sequence
        for i in range(self.sequence_length):
            if i < frame_paths_length:
                frame_path = video_frame_paths[i]
                if os.path.exists(frame_path):  # Only process if file exists
                    image = Image.open(frame_path)
                    if self.transform:
                        image = self.transform(image)
                    frames.append(image)
                else:
                    print(f"File not found: {frame_path}")
            else:
                # If there aren't enough frames, use the last frame to pad
                frames.append(frames[-1])

        # Stack frames along the time dimension
        frames = torch.stack(frames)  # Shape (sequence_length, C, H, W)
        
        label = self.labels[idx]
        return frames, label

# Function to load sequences of frames from Processed_Video_Frames
def load_data(processed_video_dir, sequence_length=30):
    video_frame_sequences = []  # Each element is a list of paths for video frames in the sequence
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
                    
                    # Include folders with less than 30 frames by padding with the last frame
                    if len(frame_paths) > 0:
                        video_frame_sequences.append(frame_paths[:sequence_length])
                        labels.append(label_folder)  # Use the folder name as the label
                    else:
                        print(f"Skipping {video_folder_path}, no frames found.")

    return video_frame_sequences, labels

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((80, 60)),  # Resize to match model input
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
])

# Load your dataset from the existing Processed_Video_Frames directory
processed_video_frames_dir = r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\Processed_Video_Frames"
video_frame_paths, labels = load_data(processed_video_frames_dir, sequence_length=30)

# Split the dataset into training and validation sets (80% train, 20% validation)
train_video_paths, val_video_paths, train_labels, val_labels = train_test_split(
    video_frame_paths, labels, test_size=0.2, random_state=42)

# Convert labels to numerical values using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(labels)  # Fit on all labels to include unseen ones
train_labels = label_encoder.transform(train_labels)
val_labels = label_encoder.transform(val_labels)

# Save the label encoder for later use
joblib.dump(label_encoder, r'C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\label_encoder.pkl')

# Create datasets and dataloaders
train_dataset = SignLanguageDataset(train_video_paths, train_labels, transform=transform, sequence_length=30)
val_dataset = SignLanguageDataset(val_video_paths, val_labels, transform=transform, sequence_length=30)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize the model
model = CNNLSTMModel(num_classes=len(label_encoder.classes_), hidden_dim=2048, num_layers=10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training the model with validation accuracy tracking
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for frames, labels in train_loader:
            if frames is None:  # Skip empty sequences
                continue
            frames, labels = frames.to(device), labels.to(device).long()  # Ensure labels are LongTensor
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track training accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_frames, val_labels in val_loader:
                if val_frames is None:  # Skip empty sequences
                    continue
                val_frames, val_labels = val_frames.to(device), val_labels.to(device).long()

                val_outputs = model(val_frames)
                _, val_predicted = torch.max(val_outputs, 1)

                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        # Print training and validation stats
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

        # Adjust learning rate using scheduler
        scheduler.step()

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=20)

# Save the trained model
torch.save(model.state_dict(), r'C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\sign_language_cnn_lstm_model.pth')
