import os
import cv2 
import logging
from tqdm import tqdm 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

labels_base_folder = "/Users/mathieu/Documents/SoccerNet/sn-gamestate/new/labels"
images_base_folder = "/Users/mathieu/Documents/SoccerNet/sn-gamestate/new/images"
output_base_folder = "/Users/mathieu/Documents/SoccerNet/sn-gamestate/new/murge"
train_labels_folder = os.path.join(labels_base_folder, "train")
test_labels_folder = os.path.join(labels_base_folder, "test")
train_images_folder = os.path.join(images_base_folder, "train")
test_images_folder = os.path.join(images_base_folder, "test")
train_output_folder = os.path.join(output_base_folder, "train")
test_output_folder = os.path.join(output_base_folder, "test")

os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)

def draw_bounding_boxes(image_path, label_path, output_path):

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        color = (0, 255, 0) 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(output_path, image)

def process_folder(images_folder, labels_folder, output_folder):
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(labels_folder) if f.endswith('.txt')], key=lambda x: int(os.path.splitext(x)[0]))

    if len(image_files) != len(label_files):
        logging.warning("The number of images and labels do not match!")

    for image_file, label_file in tqdm(zip(image_files, label_files), total=len(image_files), desc=f"Processing {os.path.basename(images_folder)}"):
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, label_file)
        output_path = os.path.join(output_folder, image_file)
        draw_bounding_boxes(image_path, label_path, output_path)

# Process train folders
logging.info("Processing train folders")
process_folder(train_images_folder, train_labels_folder, train_output_folder)
logging.info("Finished processing train folders")

# Process test folders
logging.info("Processing test folders")
process_folder(test_images_folder, test_labels_folder, test_output_folder)
logging.info("Finished processing test folders")

logging.info("Image annotation completed.")
