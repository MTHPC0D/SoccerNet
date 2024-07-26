# test.py
from ultralytics import YOLO
import cv2
import os

def test_model(model_path: str, test_images_dir: str, results_dir: str):
    # Load the model with "best" weights
    model = YOLO(model_path)

    # Create the folder to save the results
    os.makedirs(results_dir, exist_ok=True)

    # Valid image file extensions
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Iterate over the test images
    for img_name in os.listdir(test_images_dir):
        if not img_name.lower().endswith(valid_extensions):
            print(f"Skipping non-image file {img_name}")
            continue

        img_path = os.path.join(test_images_dir, img_name)
        img = cv2.imread(img_path)

        # Check if the image is loaded correctly
        if img is None:
            print(f"Failed to load image {img_path}")
            continue

        # Perform inference
        results = model(img)

        # Annotate and save each image
        for result in results:
            # Annotate the image with the results
            annotated_img = result.plot()

            # Make sure to add a valid file extension
            save_path = os.path.join(results_dir, f'{os.path.splitext(img_name)[0]}_annotated.jpg')
            cv2.imwrite(save_path, annotated_img)

            print(f"Saved annotated image to {save_path}")
