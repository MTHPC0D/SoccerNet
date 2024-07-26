import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing the images
image_folder = '/Users/mathieu/Programs/soccer-net/datasets/roboflow_football_shirt/test/images'

# List to store information about the images
image_data = []

# Iterate over files in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        filepath = os.path.join(image_folder, filename)
        
        # Read the image
        image = cv2.imread(filepath)
        if image is None:
            continue
        
        # Get the dimensions of the image
        height, width, channels = image.shape
        
        # Add the information to the list
        image_data.append({
            'filename': filename,
            'width': width,
            'height': height,
            'channels': channels
        })

# Convert the list to a DataFrame
df = pd.DataFrame(image_data)

# Display the first few rows of the DataFrame
print(df.head())

# Descriptive statistics on image dimensions
print(df.describe())

# Distribution of image widths and heights
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(df['width'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Image Widths')
plt.xlabel('Width (pixels)')
plt.ylabel('Number of Images')

plt.subplot(1, 2, 2)
plt.hist(df['height'], bins=30, color='green', alpha=0.7)
plt.title('Distribution of Image Heights')
plt.xlabel('Height (pixels)')
plt.ylabel('Number of Images')

plt.tight_layout()
plt.show()

# Distribution of aspect ratios (width/height)
df['aspect_ratio'] = df['width'] / df['height']

plt.figure(figsize=(6, 6))
plt.hist(df['aspect_ratio'], bins=30, color='purple', alpha=0.7)
plt.title('Distribution of Image Aspect Ratios')
plt.xlabel('Aspect Ratio (width/height)')
plt.ylabel('Number of Images')

plt.show()
