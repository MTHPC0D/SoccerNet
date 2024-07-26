import os
import json
import shutil

source_dir_train = 'datasets/umeros_joueurs/jersey-2023/train/images'
source_dir_test = 'datasets/umeros_joueurs/jersey-2023/test/images'
gt_train_file = 'datasets/umeros_joueurs/jersey-2023/train/train_gt.json'
gt_test_file = 'datasets/umeros_joueurs/jersey-2023/test/test_gt.json'
dest_dir = 'datasets/new_snet'

# Create the destination directories
for split in ['train', 'test']:
    for i in range(100):
        os.makedirs(os.path.join(dest_dir, split, str(i)), exist_ok=True)

def move_images(source_dir, gt_file, dest_dir, split):
    # Read the ground truth file
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    # Iterate over player IDs and their corresponding numbers
    for player_id, player_number in gt_data.items():
        if player_number == -1:
            continue  # Skip entries with -1
        
        player_dir = os.path.join(source_dir, player_id)
        if not os.path.exists(player_dir):
            print(f"Directory {player_dir} does not exist.")
            continue

        # Copy the images to the directory corresponding to the player number
        dest_player_dir = os.path.join(dest_dir, split, str(player_number))
        for img_file in os.listdir(player_dir):
            src_img_path = os.path.join(player_dir, img_file)
            dest_img_path = os.path.join(dest_player_dir, img_file)
            shutil.copy(src_img_path, dest_img_path)
            print(f"Copied {src_img_path} to {dest_img_path}")

# Convert training data
move_images(source_dir_train, gt_train_file, dest_dir, 'train')

# Convert test data
move_images(source_dir_test, gt_test_file, dest_dir, 'test')

print("Conversion completed.")
