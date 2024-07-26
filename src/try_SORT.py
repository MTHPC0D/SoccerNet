import cv2
import torch
import numpy as np
from sort import Sort
from keras.models import load_model
from ultralytics import YOLO
import os

player_detector = YOLO('/Users/mathieu/Programs/soccer-net/model/best.pt')
number_detector = YOLO('/Users/mathieu/Programs/soccer-net/model/detection_with_yolo_nb.pt')
cnn_model = load_model('/Users/mathieu/Programs/soccer-net/model/fine_best_cnn.hdf5')

# Function to classify numbers with your CNN model
def classify_number(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (32, 32))  # Resize to 32x32
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=-1)  # Add the channel for grayscale
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    sorted_indices = np.argsort(predictions[0])[::-1]  # Indices of predictions sorted by confidence

    # Ensure the number 0 is not used
    if sorted_indices[0] == 0 and len(sorted_indices) > 1:
        return sorted_indices[1], predictions[0][sorted_indices[1]]
    return sorted_indices[0], predictions[0][sorted_indices[0]]

video_path = '/Users/mathieu/Programs/soccer-net/input_videos/video2.mp4'
cap = cv2.VideoCapture(video_path)

output_path = os.path.join('/Users/mathieu/Programs/soccer-net/output_videos/TOTAL/video_definitive.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Try different codecs until initialization is successful
codecs = ['mp4v', 'XVID', 'MJPG', 'DIVX', 'X264']
out = None
for codec in codecs:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if out.isOpened():
        print(f"Codec {codec} used for video writing.")
        break
    else:
        print(f"Error: Codec {codec} not supported.")
        out.release()
        out = None

# Check if video writer was properly initialized
if out is None or not out.isOpened():
    print("Error: Unable to initialize video writer with available codecs.")
    cap.release()
    exit()

tracker = Sort()

# Dictionary to map track IDs to jersey numbers
track_id_to_number = {}
assigned_numbers = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    player_results = player_detector(frame)

    # Extract bounding boxes from results
    player_boxes = []
    for result in player_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            player_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])

    # Track players
    trackers = tracker.update(np.array(player_boxes))

    for d in trackers:
        x1, y1, x2, y2, track_id = map(int, d)
        if x1 < x2 and y1 < y2:  # Ensure coordinates are valid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Detect numbers in the player's ROI
            roi = frame[y1:y2, x1:x2]
            if roi.shape[0] > 0 and roi.shape[1] > 0:  # Ensure ROI has valid dimensions
                number_results = number_detector(roi)

                for result in number_results:
                    boxes = result.boxes
                    for box in boxes:
                        nx1, ny1, nx2, ny2 = box.xyxy[0]
                        if nx1 < nx2 and ny1 < ny2:  # Ensure coordinates are valid
                            nx1, ny1, nx2, ny2 = map(int, [nx1, ny1, nx2, ny2])
                            cv2.rectangle(roi, (nx1, ny1), (nx2, ny2), (0, 255, 0), 2)

                            # Extract the number sub-image
                            number_roi = roi[ny1:ny2, nx1:nx2]
                            # Classify the number with the CNN model
                            detected_number, confidence = classify_number(number_roi, cnn_model)
                            
                            # Assign a number among the top predictions
                            if detected_number != 0 and detected_number <= 99:  # Ensure the number is not 0 and does not exceed 99
                                cv2.putText(roi, f'{detected_number}', (nx1, ny1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                track_id_to_number[track_id] = detected_number

            # Display the ID or jersey number only if a number was detected
            if track_id in track_id_to_number:
                display_id = track_id_to_number[track_id]
                if display_id != 0 and display_id <= 99:  # Ensure the ID is not 0 and does not exceed 99
                    cv2.putText(frame, f'{display_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
