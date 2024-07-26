import os
import cv2
import shutil
from PIL import Image
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

source_dir = 'datasets/football_shirt/train/images'
dest_dir = 'datasets/number_dataset/train' 

for i in range(100):
    os.makedirs(os.path.join(dest_dir, str(i)), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'poubelle'), exist_ok=True)
os.makedirs(os.path.join(dest_dir, 'cote'), exist_ok=True)

# Get the list of image files
image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

class ImageSorterApp:
    def __init__(self, master, image_files):
        self.master = master
        self.image_files = image_files
        self.image_index = 0

        self.master.title("Image Sorter")
        
        # Create a label to display the image
        self.image_label = tk.Label(master)
        self.image_label.pack()

        # Create an entry for user input
        self.entry = tk.Entry(master)
        self.entry.pack()
        self.entry.bind('<Return>', self.process_entry)

        # Display the first image
        self.display_image()

    def display_image(self):
        if self.image_index < len(self.image_files):
            image_path = os.path.join(source_dir, self.image_files[self.image_index])
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            
            # Enlarge the image
            img = cv2.resize(img, (400, 400))
            
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.image_label.config(image=img)
            self.image_label.image = img
        else:
            print("All images processed.")
            self.master.quit()

    def process_entry(self, event):
        entry = self.entry.get().strip()
        self.entry.delete(0, tk.END)

        if self.image_index < len(self.image_files):
            image_file = self.image_files[self.image_index]
            image_path = os.path.join(source_dir, image_file)

            if entry.isdigit() and 0 <= int(entry) <= 99:
                target_dir = os.path.join(dest_dir, entry)
            elif entry.lower() == 'p':
                target_dir = os.path.join(dest_dir, 'poubelle')
            elif entry.lower() == 'c':
                target_dir = os.path.join(dest_dir, 'cote')
            else:
                print("Invalid entry, image marked as trash.")
                target_dir = os.path.join(dest_dir, 'poubelle')

            # Move the image to the target directory
            shutil.move(image_path, os.path.join(target_dir, image_file))

            # Move to the next image
            self.image_index += 1
            self.display_image()

# Create the Tkinter window
root = tk.Tk()
app = ImageSorterApp(root, image_files)
root.mainloop()
