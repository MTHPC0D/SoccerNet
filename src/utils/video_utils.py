import os
import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        raise ValueError("The video frames list is empty")
    
    height, width, layers = output_video_frames[0].shape
    codecs = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi')
    ]
    
    for codec, ext in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        full_output_path = os.path.splitext(output_video_path)[0] + ext
        out = cv2.VideoWriter(full_output_path, fourcc, 24, (width, height))
        
        if not out.isOpened():
            print(f"Codec {codec} failed to create the video writer.")
            continue
        
        for frame in output_video_frames:
            out.write(frame)
        out.release()

        if os.path.exists(full_output_path):
            print(f"Video successfully saved using codec: {codec} at {full_output_path}")
            return

    raise IOError(f"Unable to create the output video with the tried codecs: {output_video_path}")
