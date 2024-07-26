import cv2
import os

image_folder = '/Users/mathieu/Programs/soccer-net/tracking/challenge2023/video_courte' 
video_name = '/Users/mathieu/Programs/soccer-net/input_videos/video_courte.mp4' 

# Check if the image folder exists
if not os.path.exists(image_folder):
    print(f"Error: The image folder '{image_folder}' does not exist.")
    exit()

# Get the list of image files in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png") or img.endswith(".jpeg")]
images.sort()  

# Check if there are any images in the folder
if not images:
    print(f"Error: No image files found in the folder '{image_folder}'.")
    exit()

frame = cv2.imread(os.path.join(image_folder, images[0]))
if frame is None:
    print(f"Error: Unable to read the image '{images[0]}'.")
    exit()

height, width, layers = frame.shape

fps = 25

# Ensure the output folder exists
output_folder = os.path.dirname(video_name)
if output_folder and not os.path.exists(output_folder):
    print(f"Error: The output folder '{output_folder}' does not exist.")
    exit()

# Try different codecs
codecs = ['mp4v', 'XVID', 'MJPG']
video = None

for codec in codecs:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    if video.isOpened():
        print(f"Video successfully created using the codec '{codec}'.")
        break

# Check if the VideoWriter object was created
if not video or not video.isOpened():
    print(f"Error: Unable to create the video at '{video_name}'.")
    exit()

# Read each image and add it to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to read the image '{image}'.")
        continue
    video.write(img)

# Release the VideoWriter object
video.release()

print(f"Video successfully created and saved at: {video_name}")
