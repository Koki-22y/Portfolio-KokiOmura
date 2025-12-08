import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import confusion_matrix
import seaborn as sns

hiragana_location = 'hiragana_images'
st_image_size = (32, 32)


def load_images_from_folder(folder, target_size):
    images = []
    labels = []
    label_map = {}  # Dictionary to map labels to unique integers
    label_count = 0

    # Get the list of filenames and sort them
    filenames = sorted(os.listdir(folder))

    for filename in filenames:
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            # Load the image
            img = cv2.imread(path)
            # Resize the image
            img = cv2.resize(img, target_size)
            # Normalize the pixel values to be between 0 and 1
            img = img / 255.0
            # Extract label from the filename
            label = filename.split('0')[0].split('1')[0].split('2')[0].split('3')[0].split('4')[0].split('5')[0].split(
                '6')[0].split('7')[0].split('8')[0].split('9')[0]
            # If the label is not in the label map, assign it a unique integer
            if label not in label_map:
                label_map[label] = label_count
                label_count += 1
            images.append(img)
            labels.append(label_map[label])

            # Data Augmentation - Horizontal Flip
            images.append(cv2.flip(img, 1))
            labels.append(label_map[label])

    return np.array(images), np.array(labels), label_map


# Resize images and load labels
X_hiragana, y_hiragana, label_map = load_images_from_folder(hiragana_location, st_image_size)

# Check if any images were loaded
if len(X_hiragana) > 0:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_hiragana, y_hiragana, test_size=0.2, random_state=42)

    # Define data augmentation parameters
    # Define data augmentation parameters
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,  # Rotate images randomly by up to 30 degrees
        width_shift_range=0.1,  # Shift images horizontally by up to 10% of the width
        height_shift_range=0.1,  # Shift images vertically by up to 10% of the height
        shear_range=0.2,  # Shear transformations with a maximum shear intensity of 0.2
        zoom_range=0.2,  # Zoom in or out on images by up to 20%
        horizontal_flip=True,  # Flip images horizontally
        fill_mode='nearest'  # Fill points outside the input boundaries with the nearest pixel
    )

    # Define simplified CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(label_map), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
                  metrics=['accuracy'])

    # Adjust Learning Rate Dynamically
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Train the model
    epochs = 60
    batch_size = 16

    # Train the model with augmented data
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size,  # Number of batches per epoch
                        epochs=epochs,
                        validation_data=(X_test, y_test),
                        callbacks=[reduce_lr])

    # Save the trained model
    model.save('hiragana_model.keras')

    # Plot training and validation loss
    plt.figure(figsize=(5, 5))
    plt.plot(history.history['loss'], label='Train Loss', color='orange')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # Print test accuracy
    print("Test Accuracy:", test_accuracy)

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(y_test, predicted_classes)

    sns.heatmap(conf_matrix, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title('Confusion Matrix')
    plt.show()

else:
    print("No images loaded from the folder.")
