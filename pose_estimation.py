from ultralytics import YOLO
import cv2
import numpy as np
import os
from DataReader import convert_mov_to_frames
from Visualizer import annotate_pose_on_input_and_save_as_mp4

movie_name = 'IMG_1872'
video_path = f'/home/hazel/Datasets/image-online/{movie_name}.MOV'
img_folder = f'/home/hazel/Datasets/image-online/{movie_name}_frames'
pose_folder = f'/home/hazel/Datasets/image-online/{movie_name}_pose'
output_movie_path = f'/home/hazel/Datasets/image-online/{movie_name}_pose.mp4'
segmentation_folder = f'/home/hazel/Datasets/image-online/{movie_name}_segmentation'


if __name__ == "__main__":
    convert_mov_to_frames(video_path, img_folder)  # convert video to frames
    # load YOLO pose model
    model = YOLO('yolo11x-pose.pt')  
    model_segmentation = YOLO("yolo11n-seg.pt")  # load YOLO segmentation model
    annotate_pose_on_input_and_save_as_mp4(model, model_segmentation, 
                                           img_folder, pose_folder,
                                             output_movie_path,segmentation_folder)  # annotate frames with pose estimation



