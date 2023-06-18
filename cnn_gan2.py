import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate

# Define the latent dimension for the generator
latent_dim = 100

# Load the dataset
data_path_1 = "C:\\Users\\HI\\OneDrive\\Desktop\\cancer_detect\\HAM10000_images_part_1"
data_path_2 = "C:\\Users\\HI\\OneDrive\\Desktop\\cancer_detect\\HAM10000_images_part_2"
metadata_path = "c:\\Users\\HI\\OneDrive\\Desktop\\cancer_detect\\HAM10000_metadata.csv"

# Load the metadata
metadata = pd.read_csv(metadata_path)

# Load and preprocess the images in batches
batch_size = 100
num_images = metadata.shape[0]
images = []
labels = []

for i in range(0, num_images, batch_size):
    batch_metadata = metadata.iloc[i:i+batch_size]
    batch_images = []
    batch_labels = []

    for _, row in batch_metadata.iterrows():
        image_id = row['image_id']
        label = row['dx']

        # Search for the image in part 1
        image_path = os.path.join(data_path_1, f"{image_id}.jpg")
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
        else:
            # Search for the image in part 2
            image_path = os.path.join(data_path_2, f"{image_id}.jpg")
            image = cv2.imread(image_path)

        # Preprocess the image (resize, normalize, etc.)
        image = cv2.resize(image, (64, 64))
        image = image.astype('float32') / 255.0

        batch_images.append(image)
        batch_labels.append(label)

    # Convert the batch data to NumPy arrays
    batch_images = np.array(batch_images)
    batch_labels = np.array(batch_labels)

    # Append the batch data to the main arrays
    images.append(batch_images)
    labels.append(batch_labels)

# Concatenate the batches
images = np.concatenate(images, axis=0)
labels = np.concatenate(labels, axis=0)

# Perform label encoding
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)
np.save("label_encoder_classes3.npy", label_encoder.classes_)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels_encoded, test_size=0.2, random_state=42
)

# Convert the labels to one-hot encoded vectors
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Build the CNN model
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dense(num_classes, activation='softmax'))

cnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train the CNN model
cnn_model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# Define the generator model
generator = Sequential()
generator.add(Dense(128, activation='relu', input_dim=latent_dim))
generator.add(Dense(64 * 64 * 3, activation='tanh'))
generator.add(Reshape((64, 64, 3)))

# Define the discriminator model
discriminator = Sequential()
discriminator.add(Flatten(input_shape=(64, 64, 3)))
discriminator.add(Dense(128, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))

# Compile the discriminator model
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Combine the generator and discriminator into a GAN
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan_model = Model(gan_input, gan_output)

# Compile the GAN model
gan_model.compile(optimizer='adam', loss='binary_crossentropy')

# Train the GAN
epochs = 10
batch_size = 64

for epoch in range(epochs):
    for _ in range(len(train_images) // batch_size):
        # Train the discriminator
        real_images = train_images[np.random.randint(0, train_images.shape[0], batch_size)]
        real_labels = np.ones((batch_size, 1))

        latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        generated_images = generator.predict(latent_vectors)
        generated_labels = np.zeros((batch_size, 1))

        discriminator_loss_real = discriminator.train_on_batch(real_images, real_labels)
        discriminator_loss_generated = discriminator.train_on_batch(generated_images, generated_labels)

        discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_generated)

        # Train the generator
        latent_vectors = np.random.normal(size=(batch_size, latent_dim))
        misleading_labels = np.ones((batch_size, 1))

        generator_loss = gan_model.train_on_batch(latent_vectors, misleading_labels)

    print(f"Epoch {epoch + 1}/{epochs}, Discriminator Loss: {discriminator_loss}, Generator Loss: {generator_loss}")

# Save the combined model
gan_model.save("cnn_gan_model2.h5")

# Evaluate the GAN model
test_latent_vectors = np.random.normal(size=(len(test_images), latent_dim))
generated_images = generator.predict(test_latent_vectors)
discriminator_predictions = discriminator.predict(generated_images)

# Assuming binary classification (real or fake)
discriminator_predictions = discriminator_predictions > 0.5
accuracy = np.mean(discriminator_predictions == 0)

print(f"GAN Model Accuracy: {accuracy}")
