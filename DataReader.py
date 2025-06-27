
import os
import cv2
from tqdm import tqdm


def convert_mov_to_frames(video_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    # Loop through each frame and save it as an image
    for frame_num in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break  # Break if there are no more frames
        # Save the frame as an image
        frame_filename = os.path.join(output_path, f"frame_{frame_num:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
    # Release the video capture object
    cap.release()
    print(f"Frames saved to {output_path}")

