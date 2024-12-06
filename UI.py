# import tkinter as tk
# from tkinter import messagebox
# import cv2
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image, ImageTk
# import numpy as np
# import pyttsx3  # For audio output
# import threading
# from sklearn.preprocessing import LabelEncoder
# import joblib

# # Load the trained model
# class SignLanguageCNN(nn.Module):
#     def __init__(self, num_classes=101):  # Set num_classes to 101
#         super(SignLanguageCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.flattened_size = 64 * 20 * 15  # Adjust based on input size after conv layers
#         self.fc1 = nn.Linear(self.flattened_size, 128)
#         self.fc2 = nn.Linear(128, num_classes)  # Use num_classes=101

#     def forward(self, x):
#         x = self.pool(nn.ReLU()(self.conv1(x)))
#         x = self.pool(nn.ReLU()(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = nn.ReLU()(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Load the model and LabelEncoder
# model = SignLanguageCNN(num_classes=101)  # Adjust number of classes as per your dataset
# model.load_state_dict(torch.load(r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\sign_language_model_2.pth"))
# model.eval()  # Set the model to evaluation mode

# # Load the LabelEncoder
# label_encoder = joblib.load(r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\label_encoder.pkl")

# # Define transformations for input images
# transform = transforms.Compose([
#     transforms.Resize((80, 60)),  # Resize to match model input
#     transforms.ToTensor(),
# ])

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Function to perform sign detection
# def detect_sign(frame):
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     image = Image.fromarray(image)
#     image = transform(image).unsqueeze(0)  # Add batch dimension

#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)

#     # Get the predicted class
#     predicted_label = label_encoder.inverse_transform([predicted.item()])[0]  # Decode label

#     # Speak the detected sign
#     engine.say(predicted_label)
#     engine.runAndWait()

#     return predicted_label

# # Function to update the video feed in the Tkinter window
# def update_frame():
#     ret, frame = cap.read()
#     if ret:
#         detected_sign = detect_sign(frame)
#         # Update the message box with the detected sign
#         detected_sign_var.set(f"Detected: {detected_sign}")

#         # Convert frame to ImageTk format to display in Tkinter
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#         imgtk = ImageTk.PhotoImage(image=img)
#         video_label.imgtk = imgtk
#         video_label.configure(image=imgtk)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         root.after(10, update_frame)  # Schedule next frame update

# # Function to run video processing in a separate thread
# def start_detection():
#     thread = threading.Thread(target=update_frame)
#     thread.daemon = True  # Ensure the thread closes when the main window closes
#     thread.start()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Create the main UI window
# root = tk.Tk()
# root.title("Sign Language Detection")
# root.geometry("800x600")

# # Label for showing the video
# video_label = tk.Label(root)
# video_label.pack(pady=20)

# # Variable and label for detected sign
# detected_sign_var = tk.StringVar()
# detected_sign_label = tk.Label(root, textvariable=detected_sign_var, font=("Helvetica", 16))
# detected_sign_label.pack(pady=20)

# # Add a label for instructions
# instruction_label = tk.Label(root, text="Press 'q' to exit", font=("Helvetica", 16))
# instruction_label.pack(pady=10)

# # Start the video detection in a separate thread
# start_detection()

# # Start the Tkinter main loop
# root.mainloop()

# # Release resources when done
# cap.release()
# cv2.destroyAllWindows()


# import tkinter as tk
# from tkinter import messagebox
# import cv2
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image, ImageTk
# import numpy as np
# import pyttsx3  # For audio output
# import threading
# from queue import Queue
# from sklearn.preprocessing import LabelEncoder
# import joblib
# from collections import deque

# # Device configuration (Use GPU if available)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load the trained model
# class SignLanguageCNN(nn.Module):
#     def __init__(self, num_classes=101):  # Set num_classes to 101
#         super(SignLanguageCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.flattened_size = 64 * 20 * 15  # Adjust based on input size after conv layers
#         self.fc1 = nn.Linear(self.flattened_size, 128)
#         self.fc2 = nn.Linear(128, num_classes)  # Use num_classes=101

#     def forward(self, x):
#         x = self.pool(nn.ReLU()(self.conv1(x)))
#         x = self.pool(nn.ReLU()(self.conv2(x)))
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = nn.ReLU()(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Load the model and move it to the GPU
# model = SignLanguageCNN(num_classes=101)  # Adjust number of classes as per your dataset
# model.load_state_dict(torch.load(r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\sign_language_cnn_lstm_model.pth"))
# model.to(device)  # Move the model to GPU
# model.eval()  # Set the model to evaluation mode

# # Load the LabelEncoder
# label_encoder = joblib.load(r"C:\Users\Souvik Baidya\Documents\Software Engineering\Project\ISL Video Dataset\ISL_CSLRT_Corpus\ISL_CSLRT_Corpus\label_encoder.pkl")

# # Define transformations for input images
# transform = transforms.Compose([
#     transforms.Resize((80, 60)),  # Resize to match model input
#     transforms.ToTensor(),
# ])

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Queue to share frames between video capture and detection threads
# frame_queue = Queue()

# # Sliding window for stabilizing predictions
# prediction_window = deque(maxlen=5)  # Adjust window size

# # Confidence threshold for sign prediction
# confidence_threshold = 0.9

# # Function to preprocess frames (e.g., contrast enhancement)
# def preprocess_frame(frame):
#     # Convert to grayscale for better contrast and consistency
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Apply histogram equalization to enhance contrast
#     equalized = cv2.equalizeHist(gray)
#     # Convert back to 3-channel format (as the model expects RGB input)
#     equalized_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
#     return equalized_rgb

# # Function to perform sign detection with stabilization and confidence thresholding
# def detect_sign(frame):
#     # Mirror the frame
#     frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror)
    
#     # Preprocess the frame
#     processed_frame = preprocess_frame(frame)
    
#     image = Image.fromarray(processed_frame)
#     image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

#     with torch.no_grad():
#         output = model(image)
#         probabilities = torch.nn.functional.softmax(output, dim=1)  # Get probabilities
#         _, predicted = torch.max(output, 1)
#         confidence = probabilities[0][predicted.item()].item()

#     # Only consider predictions with confidence higher than the threshold
#     if confidence > confidence_threshold:
#         predicted_label = label_encoder.inverse_transform([predicted.item()])[0]
#     else:
#         predicted_label = "Low Confidence"

#     # Add prediction to the sliding window for stabilization
#     prediction_window.append(predicted_label)

#     # Get the most frequent prediction in the sliding window
#     stable_prediction = max(set(prediction_window), key=prediction_window.count)

#     # Speak the stabilized prediction
#     if stable_prediction != "Low Confidence":
#         engine.say(stable_prediction)
#         engine.runAndWait()

#     return stable_prediction

# # Function to update the video feed in the Tkinter window
# def update_frame():
#     ret, frame = cap.read()
#     if ret:
#         # Process frame for detection
#         frame_queue.put(frame)  # Put the frame into the queue for detection
        
#         # Display video in Tkinter window
#         img = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(img)
#         imgtk = ImageTk.PhotoImage(image=img)
#         video_label.imgtk = imgtk
#         video_label.configure(image=imgtk)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         root.after(10, update_frame)  # Schedule next frame update

# # Function to handle sign detection on a separate thread
# def detect_sign_thread():
#     while True:
#         # Check if there is a frame in the queue
#         if not frame_queue.empty():
#             frame = frame_queue.get()
#             detected_sign = detect_sign(frame)
#             # Update the message box with the detected sign
#             detected_sign_var.set(f"Detected: {detected_sign}")

# # Function to run video processing in a separate thread
# def start_detection():
#     # Thread for updating video frames
#     video_thread = threading.Thread(target=update_frame)
#     video_thread.daemon = True  # Ensure the thread closes when the main window closes
#     video_thread.start()

#     # Thread for sign detection
#     detection_thread = threading.Thread(target=detect_sign_thread)
#     detection_thread.daemon = True
#     detection_thread.start()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# # Create the main UI window
# root = tk.Tk()
# root.title("Sign Language Detection")
# root.geometry("800x600")

# # Label for showing the video
# video_label = tk.Label(root)
# video_label.pack(pady=20)

# # Variable and label for detected sign
# detected_sign_var = tk.StringVar()
# detected_sign_label = tk.Label(root, textvariable=detected_sign_var, font=("Helvetica", 16))
# detected_sign_label.pack(pady=20)

# # Add a label for instructions
# instruction_label = tk.Label(root, text="Press 'q' to exit", font=("Helvetica", 16))
# instruction_label.pack(pady=10)

# # Start the video detection in a separate thread
# start_detection()

# # Start the Tkinter main loop
# root.mainloop()

# # Release resources when done
# cap.release()
# cv2.destroyAllWindows()

#tensorflowjs_converter --input_format=tfjs_layers_model --output_format=tf_saved_model "C:\Users\Souvik Baidya\Downloads\my-pose-model" path/to/output_model_dir









# import os
# import tensorflow as tf
# import json
# import numpy as np
# import tkinter as tk
# from tkinter import Label, Button
# import cv2

# # Load the Teachable Machine model
# def load_teachable_machine_model():
#     model_dir = r"C:\Users\Souvik Baidya\Downloads\my-pose-model"  # Your actual path
#     with open(os.path.join(model_dir, 'model.json')) as f:
#         model_json = json.load(f)
    
#     # Create the model from model.json
#     model = tf.keras.models.model_from_json(json.dumps(model_json['modelTopology']))
    
#     # Load the weights
#     weights_path = os.path.join(model_dir, 'weights.bin')
#     weights = np.fromfile(weights_path, dtype=np.float32)
    
#     # Assign the weights manually based on the structure in model.json
#     for layer, weight_array in zip(model.layers, weights):
#         layer.set_weights(weight_array)

#     return model

# def preprocess_frame(frame):
#     frame_resized = cv2.resize(frame, (320, 240))
#     frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#     frame_normalized = frame_rgb / 255.0
#     return np.expand_dims(frame_normalized, axis=0)

# def predict_sign(model, frame):
#     predictions = model.predict(frame)
#     predicted_label = np.argmax(predictions, axis=-1)
#     confidence = np.max(predictions)
#     return predicted_label, confidence

# def map_label_to_sign(label):
#     labels = ["A LOT", "ABUSE", "ANGRY", "BED/SLEEP", "COMB", "CRY", "FINE", "HOW/WHEN/WHERE/WHO", "LOVE", "UNDERSTAND", "MEDICINE"]
#     return labels[label]

# def create_ui():
#     model = load_teachable_machine_model()

#     root = tk.Tk()
#     root.title("Sign Language Detection")
#     root.geometry("500x400")

#     label = Label(root, text="Sign Language Detection", font=("Arial", 20))
#     label.pack(pady=20)

#     start_button = Button(root, text="Start Detection", font=("Arial", 14), command=lambda: real_time_prediction(model))
#     start_button.pack(pady=20)

#     quit_button = Button(root, text="Quit", font=("Arial", 14), command=root.quit)
#     quit_button.pack(pady=10)

#     root.mainloop()

# def real_time_prediction(model):
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         preprocessed_frame = preprocess_frame(frame)
#         predicted_label, confidence = predict_sign(model, preprocessed_frame)

#         if predicted_label is not None:
#             text = f"Sign: {map_label_to_sign(predicted_label)}, Confidence: {confidence:.2f}"
#             cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#         cv2.imshow('Real-time Sign Language Detection', frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     create_ui()










# import tkinter as tk
# from tkinter import messagebox
# import tensorflow as tf
# from PIL import Image, ImageTk
# import numpy as np
# import threading
# import cv2
# import pyttsx3  # For audio output

# # Load model and labels
# def load_model():
#     model = tf.keras.models.load_model(r"C:\Users\Souvik Baidya\Downloads\converted_keras (3)\keras_model.h5")
#     with open(r"C:\Users\Souvik Baidya\Downloads\converted_keras (3)\labels.txt", 'r') as f:
#         labels = [line.strip() for line in f.readlines()]
#     return model, labels

# # Real-time video capture
# def start_video_capture():
#     global cap
#     cap = cv2.VideoCapture(0)
#     video_loop()

# # Process video frames
# def video_loop():
#     global cap
#     ret, frame = cap.read()
#     if ret:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(frame)
#         imgtk = ImageTk.PhotoImage(image=img)
#         video_label.imgtk = imgtk
#         video_label.configure(image=imgtk)

#         # Predict sign in a separate thread for performance
#         threading.Thread(target=detect_sign, args=(frame,)).start()

#     video_label.after(10, video_loop)

# # Detect sign from frame
# def detect_sign(frame):
#     resized_frame = cv2.resize(frame, (224, 224))  # Change this to your model's input size
#     normalized_frame = np.array(resized_frame) / 255.0
#     input_frame = np.expand_dims(normalized_frame, axis=0)

#     # Make predictions
#     predictions = model.predict(input_frame)
#     predicted_index = np.argmax(predictions)
#     confidence = predictions[0][predicted_index]

#     # Update UI with prediction
#     prediction_label.config(text=f"Detected: {labels[predicted_index]} (Confidence: {confidence*100:.2f}%)")
    
#     # Audio output
#     if confidence > 0.8:  # Only speak if confidence is high enough
#         engine.say(labels[predicted_index])
#         engine.runAndWait()

# # Exit application
# def on_closing():
#     cap.release()
#     root.destroy()

# # Initialize model and labels
# model, labels = load_model()

# # Set up audio engine
# engine = pyttsx3.init()

# # Set up GUI
# root = tk.Tk()
# root.title("Sign Language Detection")

# video_label = tk.Label(root)
# video_label.pack()

# prediction_label = tk.Label(root, text="Waiting for input...", font=("Arial", 14))
# prediction_label.pack()

# # Start video button
# start_button = tk.Button(root, text="Start Video", command=start_video_capture)
# start_button.pack()

# # Quit button
# quit_button = tk.Button(root, text="Quit", command=on_closing)
# quit_button.pack()

# # Start Tkinter event loop
# root.protocol("WM_DELETE_WINDOW", on_closing)
# root.mainloop()











# import tkinter as tk
# from tkinter import Label, Button
# import cv2
# import numpy as np
# import tensorflow as tf
# import threading
# from PIL import Image, ImageTk
# from googletrans import Translator

# # Load the model
# model = tf.keras.models.load_model(r'C:\Users\Souvik Baidya\Documents\Stock prediction project\Code\converted_keras (1)\keras_model.h5')

# # Load labels from labels.txt
# def load_labels(label_path):
#     with open(label_path, 'r') as f:
#         labels = f.read().splitlines()
#     return labels

# labels = load_labels(r'C:\Users\Souvik Baidya\Documents\Stock prediction project\Code\converted_keras (1)\labels.txt')

# # Translator initialization
# translator = Translator()

# # Translate the detected sign to the selected language
# def translate_text(text, target_language):
#     translated = translator.translate(text, dest=target_language)
#     return translated.text

# # Real-time detection function with integrated video display
# def detect_sign_language(video_label, canvas, language_choice):
#     cap = cv2.VideoCapture(0)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Preprocess the frame for model prediction
#         resized_frame = cv2.resize(frame, (224, 224))  # Resize for model input
#         normalized_frame = np.array(resized_frame, dtype=np.float32) / 255.0
#         normalized_frame = np.expand_dims(normalized_frame, axis=0)

#         # Prediction from the model
#         predictions = model.predict(normalized_frame)
#         predicted_label = labels[np.argmax(predictions)]
        
#         # Translate to selected language
#         translated_label = translate_text(predicted_label, language_choice)
        
#         # Display detected sign and translation
#         video_label.config(text=f"Detected: {predicted_label} ({translated_label})")

#         # Update canvas with current frame
#         cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(cv2image)
#         imgtk = ImageTk.PhotoImage(image=img)
#         canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
#         canvas.image = imgtk

#         # Limit the number of frames processed per second to reduce lag (e.g., 30 FPS)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()

# # Create the Tkinter UI
# def create_ui():
#     root = tk.Tk()
#     root.title("Sign Language Detection")

#     # Create a label for the detected sign
#     video_label = Label(root, text="Sign will be detected here", font=("Arial", 18))
#     video_label.pack(pady=10)

#     # Create a canvas to display the video feed
#     canvas = tk.Canvas(root, width=640, height=480)
#     canvas.pack()

#     # Dropdown for language selection
#     language_label = Label(root, text="Choose Output Language", font=("Arial", 14))
#     language_label.pack(pady=5)

#     language_choice = tk.StringVar(value='en')  # Default language is English
#     language_dropdown = tk.OptionMenu(root, language_choice, 'en', 'es', 'fr', 'de', 'nl', 'it', 'pt', 'hi', 'ja', 'zh')  # Add more languages if needed
#     language_dropdown.pack(pady=10)

#     # Start detection button
#     start_button = Button(root, text="Start Video Detection", font=("Arial", 14), 
#                           command=lambda: threading.Thread(target=detect_sign_language, args=(video_label, canvas, language_choice.get())).start())
#     start_button.pack(pady=10)

#     # Quit button
#     quit_button = Button(root, text="Quit", font=("Arial", 14), command=root.quit)
#     quit_button.pack(pady=10)

#     root.mainloop()

# # Run the UI
# if __name__ == "__main__":
#     create_ui()













import sys
import cv2
import numpy as np
import threading
import tensorflow as tf
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtMultimedia import QSound
import pyttsx3  # For audio output

# Load model and labels
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\Souvik Baidya\Downloads\converted_keras (4)\keras_model.h5")
    with open(r"C:\Users\Souvik Baidya\Downloads\converted_keras (4)\labels.txt", 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

class SignLanguageApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load model and labels
        self.model, self.labels = load_model()
        self.engine = pyttsx3.init()

        # Initialize camera
        self.cap = cv2.VideoCapture(0)

        # UI setup
        self.initUI()

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        # Window settings
        self.setWindowTitle("Sign Language Detection")
        self.setGeometry(300, 100, 800, 600)
        self.setStyleSheet("background-color: #1f1f1f; color: white;")

        # Create main layout
        layout = QVBoxLayout()

        # Create label to display video
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Create label to display predictions
        self.prediction_label = QLabel("Waiting for input...", self)
        self.prediction_label.setFont(QFont('Arial', 16))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("color: #32CD32;")
        layout.addWidget(self.prediction_label)

        # Start Video button
        self.start_button = QPushButton("Start Video", self)
        self.start_button.setFont(QFont('Arial', 14))
        self.start_button.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px 20px; border-radius: 8px;"
        )
        self.start_button.clicked.connect(self.start_video)
        layout.addWidget(self.start_button)

        # Quit button
        self.quit_button = QPushButton("Quit", self)
        self.quit_button.setFont(QFont('Arial', 14))
        self.quit_button.setStyleSheet(
            "background-color: #f44336; color: white; padding: 10px 20px; border-radius: 8px;"
        )
        self.quit_button.clicked.connect(self.close_app)
        layout.addWidget(self.quit_button)

        # Center align all buttons and labels
        button_layout = QHBoxLayout()
        button_layout.addSpacerItem(QSpacerItem(50, 10, QSizePolicy.Expanding))
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.quit_button)
        button_layout.addSpacerItem(QSpacerItem(50, 10, QSizePolicy.Expanding))
        layout.addLayout(button_layout)

        # Set main layout
        self.setLayout(layout)

    def start_video(self):
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the image to RGB for PyQt display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

            # Start a new thread for prediction to avoid UI freezing
            threading.Thread(target=self.detect_sign, args=(frame,)).start()

    def detect_sign(self, frame):
        resized_frame = cv2.resize(frame, (224, 224))  # Adjust according to model input size
        normalized_frame = np.array(resized_frame) / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Make predictions
        predictions = self.model.predict(input_frame)
        predicted_index = np.argmax(predictions)
        confidence = predictions[0][predicted_index]

        # Update UI with prediction
        self.prediction_label.setText(f"Detected: {self.labels[predicted_index]} (Confidence: {confidence * 100:.2f}%)")

        # Audio output
        if confidence > 0.8:
            self.engine.say(self.labels[predicted_index])
            self.engine.runAndWait()

    def close_app(self):
        self.cap.release()
        self.close()

def main():
    app = QApplication(sys.argv)
    window = SignLanguageApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
