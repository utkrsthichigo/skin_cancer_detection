import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Load the trained GAN model
gan_model = load_model("c:\\Users\\HI\\OneDrive\\Desktop\\cancer_detect\\cnn_gan_model2.h5")  # Replace with the path to your GAN model

# Get the generator part of the GAN model
generator_model = gan_model.layers[2]

# Function to preprocess the input image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image.astype('float32') / 255.0
    return image

# Function to make predictions on the input image
def predict_image(image, label_encoder):
    preprocessed_image = preprocess_image(image)
    prediction = generator_model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_label_index = np.argmax(prediction, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_label_index)
    return predicted_label[0], predicted_label_index[0]

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("c:\\Users\\HI\\OneDrive\\Desktop\\cancer_detect\\label_encoder_classes3.npy")  # Replace with the path to your label encoder classes file

# Verify the class indices and labels
class_indices = [4, 1, 0, 2, 3, 5]  # Indices for skin cancer classes: mel, bcc, akiec, nv, vasc, df

for index in class_indices:
    if index < len(label_encoder.classes_):
        print(f"Index: {index}, Class Label: {label_encoder.classes_[index]}")
    else:
        print(f"Invalid index: {index}")

# Function to capture an image from the camera
def capture_image():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            file_name = input("Enter the name of the image (including extension) to save: ")
            file_path = os.path.join("new_data", file_name)
            cv2.imwrite(file_path, frame)
            captured_image = cv2.imread(file_path)
            prediction = predict_image(captured_image, label_encoder)

            print("Predicted Label: ", prediction[0])
            print("Label Index: ", prediction[1])

            if prediction[1] in class_indices:
                print("Prediction: The person may have skin cancer. Label:", prediction[0])
            else:
                print("Prediction: The person does not have skin cancer.")

            break

    camera.release()
    cv2.destroyAllWindows()


def choose_existing_file():
    file_name = input("Enter the name of the file (including extension) from the 'new_data' folder: ")
    file_path = os.path.join("new_data", file_name)
    if os.path.isfile(file_path):
        captured_image = cv2.imread(file_path)
        prediction = predict_image(captured_image, label_encoder)

        print("Predicted Label: ", prediction[0])
        print("Label Index: ", prediction[1])

        if prediction[1] in class_indices:
            print("Prediction: The person may have skin cancer. Label:", prediction[0])
        else:
            print("Prediction: The person does not have skin cancer.")
    else:
        print("File not found.")


def choose_file_by_path():
    file_path = input("Enter the path of the image file: ")
    if os.path.isfile(file_path):
        captured_image = cv2.imread(file_path)
        prediction = predict_image(captured_image, label_encoder)

        print("Predicted Label: ", prediction[0])
        print("Label Index: ", prediction[1])

        if prediction[1] in class_indices:
            print("Prediction: The person may have skin cancer. Label:", prediction[0])
        else:
            print("Prediction: The person does not have skin cancer.")
    else:
        print("File not found.")


while True:
    print("Skin Cancer Detection")
    print("1. Capture an image using the camera")
    print("2. Choose an existing image file")
    print("3. Choose an image file by entering its path")
    print("0. Exit")
    choice = input("Enter your choice: ")

    if choice == "1":
        capture_image()
    elif choice == "2":
        choose_existing_file()
    elif choice == "3":
        choose_file_by_path()
    elif choice == "0":
        break
    else:
        print("Invalid choice. Please try again.")
