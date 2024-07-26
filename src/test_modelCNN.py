import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

model_path = '/Users/mathieu/Programs/soccer-net/model/fine_best_cnn.hdf5'  
model = load_model(model_path)

image_folder = '/Users/mathieu/Programs/soccer-net/src/test/num'  

images = []
image_names = []

# Expected size by the model (32x32 pixels, in grayscale)
target_size = (32, 32)

# Iterate over all images in the folder
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    if img_path.endswith('.jpg') or img_path.endswith('.png'): 
        img = image.load_img(img_path, target_size=target_size, color_mode='grayscale') 
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0  
        images.append(img_array)
        image_names.append(img_name)

images = np.vstack(images)

predictions = model.predict(images)

# Display the results
for i, img_array in enumerate(images):
    plt.imshow(img_array.squeeze(), cmap='gray')
    plt.title(f'Prediction: {np.argmax(predictions[i])}, Image: {image_names[i]}')
    plt.show()
