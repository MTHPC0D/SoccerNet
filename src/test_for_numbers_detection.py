from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import imutils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

test_images_path = '/Users/mathieu/Programs/soccer-net/datasets/roboflow_football_shirt/test/images'

model = YOLO('/Users/mathieu/Programs/soccer-net/model/detection_with_yolo_nb.pt')

testFilenames = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.endswith('.jpg')]

fig, ax = plt.subplots(2, 5, figsize=(20, 10))
ax = ax.ravel()

for i, imagePath in enumerate(testFilenames[10:20]):
    image = load_img(imagePath, target_size=(224, 224))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    results = model.predict(source=imagePath, save=False)

    # Load the input image (in OpenCV format), resize it, and get its dimensions
    image_cv = cv2.imread(imagePath)
    original_height, original_width = image_cv.shape[:2]

    ax[i].imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))

    # Draw the predicted bounding box on the image
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax[i].add_patch(rect)
    
    ax[i].set_title(f"Image {i + 10}")
    ax[i].axis('off')

plt.tight_layout()
plt.show()
