import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

input_shape = (32, 32, 3)

# Load the VGG16 model with the provided weights
base_model = VGG16(weights=None, include_top=False, input_shape=input_shape)
base_model.load_weights('src/data/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Add custom top layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(100, activation='softmax')(x) 

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    # Convert the image to RGB by stacking the single channel three times
    image = np.stack((image,)*3, axis=-1)
    image = image.astype('float32') / 255
    # Expand the dimensions to match the model input
    image = np.expand_dims(image, axis=0)
    return image

def evaluate_model(model, test_images, test_labels):
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {accuracy}')

test_images = np.random.rand(200, 32, 32, 3)
test_labels = np.random.randint(0, 100, 200)   

evaluate_model(model, test_images, test_labels)

# Process and predict an example image
image_path = 'football_shirt/train/images/24_jpg.rf.d7fb736d3b2dbf1da7ce50f6cbac3b61.jpg'
processed_image = preprocess_image(image_path)

predictions = model.predict(processed_image)
predicted_class = np.argmax(predictions, axis=1)
print(f'Predicted number: {predicted_class[0]}')
