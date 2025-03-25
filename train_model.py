import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Path where the images of dogs and cats are saved
train_dir = r'E:\React\cat-and-dog-analyze-image\train'

img_width, img_height = 150, 150  # Images will be resized to 150x150 pixels
batch_size = 32  # Number of images being processed at a time

# Data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# rescale=1./255: Normalizes pixel values to the range [0, 1]
# validation_split=0.2: Splits the data into 80% training and 20% validation

# Training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Uses 20% of the data for validation
)

# Build the CNN model
model = Sequential([
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    # 32 filters of 3x3, ReLU activation function
    MaxPooling2D(pool_size=(2, 2)),  # Pooling

    # Second convolutional layer
    Conv2D(64, (3, 3), activation='relu'),  # 64 filters of 3x3
    MaxPooling2D(pool_size=(2, 2)),

    # Third convolutional layer
    Conv2D(128, (3, 3), activation='relu'),  # 128 filters of 3x3
    MaxPooling2D(pool_size=(2, 2)),

    # Flatten the output of the convolutional layers
    Flatten(),

    # Fully connected (dense) layer
    Dense(512, activation='relu'),  # 512 neurons, ReLU activation function

    # Output layer
    Dense(1, activation='sigmoid')  # 1 neuron (binary classification), sigmoid activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,  # Number of batches per epoch
    epochs=15,  # Number of epochs (full passes over the training data)
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size  # Number of validation batches
)

# Save the trained model
model.save('cat_dog_classifier.h5')
print("Model trained and saved successfully.")