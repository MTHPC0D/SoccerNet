# test.py
from ultralytics import YOLO
import cv2
import os

def test_model(model_path: str, test_images_dir: str, results_dir: str):
    # Charger le modèle avec les poids "best"
    model = YOLO(model_path)

    # Créer le dossier pour enregistrer les résultats
    os.makedirs(results_dir, exist_ok=True)

    # Extensions de fichiers d'image valides
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Parcourir les images de test
    for img_name in os.listdir(test_images_dir):
        if not img_name.lower().endswith(valid_extensions):
            print(f"Skipping non-image file {img_name}")
            continue

        img_path = os.path.join(test_images_dir, img_name)
        img = cv2.imread(img_path)

        # Vérifiez si l'image est correctement chargée
        if img is None:
            print(f"Failed to load image {img_path}")
            continue

        # Effectuer une inférence
        results = model(img)

        # Annoter et sauvegarder chaque image
        for result in results:
            # Annoter l'image avec les résultats
            annotated_img = result.plot()

            # Assurez-vous d'ajouter une extension de fichier valide
            save_path = os.path.join(results_dir, f'{os.path.splitext(img_name)[0]}_annotated.jpg')
            cv2.imwrite(save_path, annotated_img)

            print(f"Saved annotated image to {save_path}")
