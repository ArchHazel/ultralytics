import os
import cv2
import numpy as np


def annotate_pose_on_input_and_save_as_mp4(model, model_seg, img_folder, pose_folder, movie_path, segmentation_folder ):
    imgs_path = os.listdir(img_folder)
    imgs_path = np.sort(imgs_path)  # sort the images by name
    os.makedirs(pose_folder, exist_ok=True)  # create output folder if it doesn't exist
    os.makedirs(segmentation_folder, exist_ok=True)  # create output folder if it doesn't exist
    log_keypoints = {}
    for img_path in imgs_path:
        # if img_path != "frame_0339.jpg":
        #     continue
        results = model(os.path.join(img_folder, img_path))  # predict on image
        results_seg = model_seg(os.path.join(img_folder, img_path))  # predict on image
        h,w = results[0].orig_shape[:2]  # get original image shape
        # print(results[0].keypoints.data)
        cv2.imwrite(os.path.join(pose_folder, img_path), results[0].plot())  # save image with keypoints
        if(results_seg[0].masks is not None):
            # find index of human 
            human_indices = (results_seg[0].boxes.cls == 0).nonzero()
            if len(human_indices) == 0 or human_indices[0].numel() == 0:
                print("No human detected")
                continue
            elif human_indices[0].numel() > 1:
                print("Multiple humans detected")
                continue
            human_index = human_indices[0][0].item()
            # Convert mask to single channel image
            mask_raw = results_seg[0].masks[human_index].cpu().data.numpy().transpose(1, 2, 0)
            
            # Convert single channel grayscale to 3 channel image
            mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))

            # Get the size of the original image (height, width, channels)
            h2, w2, c2 = results_seg[0].orig_img.shape
            
            # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
            mask = cv2.resize(mask_3channel, (w2, h2))

            # Convert BGR to HSV
            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

            # Define range of brightness in HSV
            lower_black = np.array([0,0,0])
            upper_black = np.array([0,0,1])

            # Create a mask. Threshold the HSV image to get everything black
            mask = cv2.inRange(mask, lower_black, upper_black)

            # Invert the mask to get everything but black
            mask = cv2.bitwise_not(mask)

            # Apply the mask to the original image
            masked = cv2.bitwise_and(results_seg[0].orig_img, results_seg[0].orig_img, mask=mask)

            # Show the masked part of the image
            cv2.imwrite(os.path.join(segmentation_folder, img_path), masked)  # save image with keypoints

        # give me visible pose keypoints
        torso_index = [5,6,11,12]
        # print(len(results))
        # print(results[0].keypoints.data.shape)
        if results[0].keypoints.data.shape[0] ==0:
            print("No keypoints detected")
            if log_keypoints.get(img_path) is not None:
                del log_keypoints[img_path]
            continue
        is_torso_keypoinrts = results[0].keypoints.data[0][6]
        is_visible_keypoints = results[0].keypoints.data[0][:,-1] > 0.8
        keypoints = results[0].keypoints.data[0][:,:-1][is_visible_keypoints]
        idx = is_visible_keypoints.nonzero()

        x_l, y_l = keypoints[:, 0], keypoints[:, 1]
        
        log_ids = []
        for x, y, id in zip(x_l, y_l, idx):
            # check if the keypoints are on the human
            if id.item() not in torso_index:
                continue

            if mask_raw[int(y/h*384), int(x/w*640)] ==0 :
                print("Keypoints are not on the human")
                continue
            else:
                if log_keypoints.get(img_path) is  None:
                    log_keypoints[img_path] = [[int(x), int(y)]]
                else:
                    log_keypoints[img_path].append([int(x), int(y)])
                log_ids.append(id.item())
        
        if len(log_ids) < len(torso_index):
            print("Not all torso keypoints are detected")
            # log_keypoints[img_path] = None
            if log_keypoints.get(img_path) is not None:
                del log_keypoints[img_path]
    
    # save the keypoints to a file
    import json
    with open(os.path.join(pose_folder, 'keypoints.json'), 'w') as f:
        json.dump(log_keypoints, f)



    # save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frame_size = (w, h)  # set frame size to original image size
    out = cv2.VideoWriter(movie_path, fourcc, fps, frame_size)
    imgs_path = os.listdir(pose_folder)
    imgs_path = np.sort(imgs_path)  # sort the images by name
    for img_path in imgs_path:
        img = cv2.imread(os.path.join(pose_folder, img_path))
        out.write(img)  # write the image to the video
    out.release()  # release the video writer