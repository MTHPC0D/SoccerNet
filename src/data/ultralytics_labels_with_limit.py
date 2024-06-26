import os
import shutil
import logging
import random
from collections import defaultdict

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dimensions de l'image
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Limite du nombre de fichiers
MAX_FILES = 2000
TRAIN_FILES = 1600
TEST_FILES = 400

# Dossiers de base
train_folder = "/Users/mathieu/Documents/SoccerNet/sn-gamestate/tracking/train"
test_folder = "/Users/mathieu/Documents/SoccerNet/sn-gamestate/tracking/test"
output_base_folder = "/Users/mathieu/Documents/SoccerNet/sn-gamestate/new/labels"
images_output_folder = "/Users/mathieu/Documents/SoccerNet/sn-gamestate/new/images"
train_output_folder = os.path.join(output_base_folder, "train")
test_output_folder = os.path.join(output_base_folder, "test")
train_images_output_folder = os.path.join(images_output_folder, "train")
test_images_output_folder = os.path.join(images_output_folder, "test")

# Créer les dossiers de sortie s'ils n'existent pas
os.makedirs(train_output_folder, exist_ok=True)
os.makedirs(test_output_folder, exist_ok=True)
os.makedirs(train_images_output_folder, exist_ok=True)
os.makedirs(test_images_output_folder, exist_ok=True)

def normalize_coordinates(x, y, w, h, img_width, img_height):
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

def sort_lines_by_image_id(lines):
    # Fonction pour extraire le premier nombre d'une ligne
    def extract_first_number(line):
        return int(line.split(',')[0])
    
    # Trier les lignes par le premier nombre
    return sorted(lines, key=extract_first_number)

def parse_gameinfo(gameinfo_file):
    tracklet_to_class = {}
    with open(gameinfo_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('trackletID'):
                parts = line.strip().split('=')
                tracklet_id = parts[0].split('_')[1]
                class_info = parts[1].split(';')
                class_name = class_info[0].strip()
                team_info = class_info[1] if len(class_info) > 1 else ''
                
                if 'player' in class_name:
                    tracklet_to_class[tracklet_id] = 1  # Classe pour les joueurs
                elif 'ball' in class_name:
                    tracklet_to_class[tracklet_id] = 2  # Classe pour la balle
                elif 'referee' in class_name:
                    tracklet_to_class[tracklet_id] = 3  # Classe pour les arbitres
                elif 'goalkeeper' in class_name:
                    tracklet_to_class[tracklet_id] = 4  # Classe pour les gardiens de but
                else:
                    tracklet_to_class[tracklet_id] = 0  # Classe inconnue ou autre
    return tracklet_to_class

def process_file(input_file, current_index, train_files_count, test_files_count, subfolder, images_folder, tracklet_to_class):
    #logging.info(f"Processing file: {input_file}")
    image_lines = defaultdict(list)
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Trier les lignes par l'ID d'image
    lines = sort_lines_by_image_id(lines)

    # Regrouper les lignes par ID d'image
    for line in lines:
        parts = line.strip().split(',')
        image_id = int(parts[0])
        tracklet_id = parts[1]
        x = float(parts[2])
        y = float(parts[3])
        w = float(parts[4])
        h = float(parts[5])
        class_id = tracklet_to_class.get(tracklet_id, 0)  # Utiliser l'ID de classe à partir du fichier gameinfo

        x_center, y_center, width, height = normalize_coordinates(x, y, w, h, IMAGE_WIDTH, IMAGE_HEIGHT)
        image_lines[image_id].append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Sélectionner une image aléatoire parmi les IDs d'images
    selected_image_id = random.choice(list(image_lines.keys()))

    # Écrire les données regroupées dans les fichiers de labels et copier l'image sélectionnée
    if current_index > MAX_FILES:
        return current_index, train_files_count, test_files_count

    label_file = os.path.join(output_base_folder, f"{current_index:06}.txt")  # Format 6 chiffres
    with open(label_file, 'w') as label:
        label.writelines(image_lines[selected_image_id])

    # Copier l'image correspondante
    image_file_name = f"{selected_image_id:06}.jpg"  # Assuming the image files are in JPEG format
    src_image_path = os.path.join(images_folder, image_file_name)

    # Déterminer si le fichier doit aller dans train ou test
    if train_files_count < TRAIN_FILES:
        dest_label_path = os.path.join(train_output_folder, f"{current_index:06}.txt")
        dest_image_path = os.path.join(train_images_output_folder, f"{current_index:06}.jpg")
        train_files_count += 1
    else:
        dest_label_path = os.path.join(test_output_folder, f"{current_index:06}.txt")
        dest_image_path = os.path.join(test_images_output_folder, f"{current_index:06}.jpg")
        test_files_count += 1

    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dest_image_path)
    else:
        logging.warning(f"Image file {src_image_path} does not exist.")

    shutil.move(label_file, dest_label_path)
    #logging.info(f"The file {input_file} from folder {subfolder} generated the files {dest_label_path} and {dest_image_path}")
    current_index += 1

    return current_index, train_files_count, test_files_count

def process_subfolders(base_folder, start_index, train_files_count, test_files_count):
    logging.info(f"Processing subfolders in: {base_folder}")
    current_index = start_index
    subfolders = sorted(os.listdir(base_folder))
    random.shuffle(subfolders)  # Mélanger les sous-dossiers aléatoirement
    while current_index <= MAX_FILES:
        for subfolder in subfolders:
            if current_index > MAX_FILES:
                break
            subfolder_path = os.path.join(base_folder, subfolder)
            if os.path.isdir(subfolder_path):
                logging.info(f"Processing subfolder: {subfolder}")
                gt_folder = os.path.join(subfolder_path, 'gt')
                images_folder = os.path.join(subfolder_path, 'img1')  # Assumed folder for images
                gameinfo_file = os.path.join(subfolder_path, 'gameinfo.ini')

                if os.path.exists(gt_folder) and os.path.exists(gameinfo_file):
                    gt_file = os.path.join(gt_folder, 'gt.txt')
                    if os.path.exists(gt_file):
                        tracklet_to_class = parse_gameinfo(gameinfo_file)
                        current_index, train_files_count, test_files_count = process_file(
                            gt_file, current_index, train_files_count, test_files_count, subfolder, images_folder, tracklet_to_class)
    return current_index, train_files_count, test_files_count

# Process both train and test folders and merge into the labels folder
logging.info("Starting processing of train folder")
next_index = 1
train_files_count = 0
test_files_count = 0
next_index, train_files_count, test_files_count = process_subfolders(
    train_folder, next_index, train_files_count, test_files_count)
logging.info("Finished processing of train folder")

logging.info("Starting processing of test folder")
next_index, train_files_count, test_files_count = process_subfolders(
    test_folder, next_index, train_files_count, test_files_count)
logging.info("Finished processing of test folder")

logging.info("Conversion, fusion et division terminées.")
