# Sign-Language-Detection
Our project leverages advanced machine learning to detect sign language gestures and convert them directly into meaningful words and sentences, eliminating the need for letter-by-letter translation. This approach ensures faster and more natural communication between sign language users and others.

Steps to Run the Project
Clone the Repository:
Clone this repository to your local machine:

bash
Copy code
git clone <repository_url>
Verify Model Files:
Make sure the following files are present in your project directory:

keras_model.h5
labels.txt
Update File Paths in UI.py:
Open the UI.py file and update the paths for keras_model.h5 and labels.txt to match their actual locations on your system. For example:

python
Copy code
model_path = "/path/to/your/keras_model.h5"
labels_path = "/path/to/your/labels.txt"
Run the Project:
Navigate to the project directory and execute UI.py to launch the application:

bash
Copy code
python UI.py
Start Detecting Sign Language:
Use the application interface to detect and translate sign language into words or sentences in real-time.

Notes
The keras_model.h5 contains the trained machine learning model for gesture recognition.
The labels.txt maps gestures to their corresponding words or sentences.
Ensure your webcam is functional for real-time detection.
Feel free to contribute by improving the model or adding new features!
