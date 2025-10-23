from ultralytics import YOLO
import cv2
import numpy as np
import os
from DataReader import convert_mov_to_frames
from Visualizer import annotate_pose_on_input_and_save_as_mp4
from pathlib import Path

session_name_keyword = 'AnchorPointsCalibration' # 2025_03_28_17_34_53_SB-79812D
sensor_node_list = [ 'Sensor Node 12 (HAR1)' , 'Sensor Node 14 (HAR2)' ,  'Sensor Node 15 (HAR3)' , 'Sensor Node 3 (HAR4)' , 'Sensor Node 6 (HAR6)' , 'Sensor Node 8 (HAR8)'] 

def from_keyword_search_session_folder(base_folder, session_name_keyword):
    base_folder = Path(base_folder)
    session_folders = [f for f in os.listdir(base_folder) if session_name_keyword in f]
    if len(session_folders) == 0:
        raise ValueError(f"No session folder found with keyword {session_name_keyword}")
    elif len(session_folders) > 1:
        print(f"Multiple session folders found with keyword {session_name_keyword}, using the first one: {session_folders[0]}")
    return session_folders[0]


for sensor_node in sensor_node_list:

    video_folder =  Path(f"/mnt/data01/PPOSS Home of the future data collection/{sensor_node}")
    video_path = from_keyword_search_session_folder(video_folder, session_name_keyword)
    movie_name = 'rgb.avi'
    img_folder = f'{video_path}/frames'  # folder to save frames
    pose_folder = f'{video_path}/pose'
    output_movie_path = f'{video_path}/pose.mp4'
    segmentation_folder = f'{video_path}/segmentation'
    video_path = f'{video_path}/{movie_name}'  # path to the video file
    print(f"Processing {video_path}")

    convert_mov_to_frames(video_path, img_folder)  # convert video to frames
    # load YOLO pose model
    model = YOLO('yolo11x-pose.pt')  
    model_segmentation = YOLO("yolo11n-seg.pt")  # load YOLO segmentation model
    annotate_pose_on_input_and_save_as_mp4(model, model_segmentation, 
                                           img_folder, pose_folder,
                                             output_movie_path,segmentation_folder)  # annotate frames with pose estimation



