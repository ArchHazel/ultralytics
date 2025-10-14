from ultralytics import YOLO
import cv2
import numpy as np
import os
from DataReader import convert_mov_to_frames
from Visualizer import annotate_pose_on_input_and_save_as_mp4

video_path = "/home/hhan2/Datasets/PPOSS Home of the future data collection/Sensor Node 12 (HAR1)/2025_03_28_17_34_53_SB-79812D"

movie_name = 'rgb.avi'

img_folder = f'{video_path}/frames'  # folder to save frames
pose_folder = f'{video_path}/pose'
output_movie_path = f'{video_path}/pose.mp4'
segmentation_folder = f'{video_path}/segmentation'
video_path = f'{video_path}/{movie_name}'  # path to the video file


if __name__ == "__main__":
    convert_mov_to_frames(video_path, img_folder)  # convert video to frames
    # load YOLO pose model
    model = YOLO('yolo11x-pose.pt')  
    model_segmentation = YOLO("yolo11n-seg.pt")  # load YOLO segmentation model
    annotate_pose_on_input_and_save_as_mp4(model, model_segmentation, 
                                           img_folder, pose_folder,
                                             output_movie_path,segmentation_folder)  # annotate frames with pose estimation



