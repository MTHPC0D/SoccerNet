import os
import shutil

def check_and_clear_directory(directory_path):
    # Vérifier si le chemin existe et est un répertoire
    if not os.path.exists(directory_path):
        print(f"Le chemin {directory_path} n'existe pas.")
        return

    if not os.path.isdir(directory_path):
        print(f"Le chemin {directory_path} n'est pas un répertoire.")
        return

    # Liste le contenu du répertoire
    files = os.listdir(directory_path)

    # Vérifie si le répertoire est vide
    if not files:
        print(f"Le répertoire {directory_path} est déjà vide.")
    else:
        # Supprime le contenu du répertoire
        for file in files:
            file_path = os.path.join(directory_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Erreur lors de la suppression de {file_path}. Raison: {e}")
        print(f"Le contenu du répertoire {directory_path} a été supprimé.")

# Exemple d'utilisation
#check_and_clear_directory("/Users/mathieu/Documents/SoccerNet/sn-gamestate/YOLO/murge")
#check_and_clear_directory("/Users/mathieu/Documents/SoccerNet/sn-gamestate/YOLO/labels")
#check_and_clear_directory("/Users/mathieu/Documents/SoccerNet/sn-gamestate/YOLO/images")