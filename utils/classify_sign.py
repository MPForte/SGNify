import json
import pickle

import numpy as np
import scipy

from utils.rps import find_rps
import os


import json
import pickle
import os

import numpy as np
import scipy

from utils.rps import find_rps


def compute_sign_class(*, result_folder, openpose_folder, segment_path, sign_class_path):
    # First, let's find the actual frame numbers available
    keypoint_files = sorted([f for f in openpose_folder.glob("*_keypoints.json")])
    
    if not keypoint_files:
        print(f"Warning: No keypoint files found in {openpose_folder}")
        sign_class_path.write_text("0a")
        return
    
    # Extract frame numbers from filenames
    frame_numbers = []
    for kp_file in keypoint_files:
        frame_num = int(os.path.basename(kp_file).split('_')[0])
        frame_numbers.append(frame_num)
    
    # Make sure we have at least some frames
    if not frame_numbers:
        print("No valid frame numbers found")
        sign_class_path.write_text("0a")
        return
    
    # We'll use the actual list of frame numbers
    num_frames = len(frame_numbers)

    # Calculate indices for the segmentation points based on available frames
    # We want to divide the available frames into roughly four segments
    t1_idx = 0  # First frame
    t2_idx = num_frames // 4  # ~25% through the frames
    t3_idx = 3 * num_frames // 4  # ~75% through the frames
    t4_idx = num_frames - 1  # Last frame

    # Make sure we have valid indices (in case of very few frames)
    t2_idx = max(0, min(t2_idx, num_frames - 1))
    t3_idx = max(t2_idx, min(t3_idx, num_frames - 1))

    # Get the actual frame numbers at these indices
    new_segment = {
        "t1": frame_numbers[t1_idx],
        "t2": frame_numbers[t2_idx],
        "t3": frame_numbers[t3_idx],
        "t4": frame_numbers[t4_idx]
    }
    
    # Save the updated segmentation
    with segment_path.open('w') as f:
        json.dump(new_segment, f)
    
    print(f"Updated segmentation to: {new_segment}")
    
    # Now run find_rps with the updated segmentation
    find_rps(result_path=result_folder, sign_class="-1", out_prefix="sign_classification", segment_path=segment_path)
    
    # Collect keypoints for frames in the segment range
    keypoints = []
    for j in frame_numbers:  # Use actual frame numbers instead of range
        file_name = f"{j:03d}_keypoints.json"
        with openpose_folder.joinpath(file_name).open() as json_file:
            json_data = json.load(json_file)
        if json_data["people"]:
            keypoints.append(json_data["people"][0]["pose_keypoints_2d"])
    
    if not keypoints:
        print("Warning: No valid keypoints found")
        sign_class_path.write_text("0a")
        return
    
    # Process keypoints as before
    x, y, c = np.transpose(np.reshape(keypoints, (-1, 25, 3)), (2, 1, 0))
    c_r = ~np.isclose(c[4], 0)
    c_l = ~np.isclose(c[7], 0)
    h = np.median(y[1] - y[0])
    y_r = y[4][c_r] / h
    y_l = y[7][c_l] / h

    v1 = min(
        np.ptp(y_r) if len(y_r) else 0,
        np.ptp(y_l) if len(y_l) else 0
    )

    base_folder = result_folder.joinpath("rps", "sign_classification")
    try:
        with base_folder.joinpath("ref_right_1.pkl").open("rb") as file:
            right_1 = pickle.load(file)["right_hand_pose"][0]
        with base_folder.joinpath("ref_right_2.pkl").open("rb") as file:
            right_2 = pickle.load(file)["right_hand_pose"][0]
        with base_folder.joinpath("ref_left_1.pkl").open("rb") as file:
            left_1 = pickle.load(file)["left_hand_pose"][0]
        with base_folder.joinpath("ref_left_2.pkl").open("rb") as file:
            left_2 = pickle.load(file)["left_hand_pose"][0]
    except Exception as e:
        print(f"Warning: Could not load hand pose data: {e}")
        right_1 = np.zeros((12,))
        right_2 = np.zeros((12,))
        left_1 = np.zeros((12,))
        left_2 = np.zeros((12,))

    try:
        v3 = scipy.spatial.distance.cosine(right_1, left_1)
        if np.isnan(v3):
            v3 = 0
    except Exception:
        v3 = 0

    try:
        v5 = max(
            scipy.spatial.distance.cosine(right_1, right_2),
            scipy.spatial.distance.cosine(left_1, left_2)
        )
        if np.isnan(v5):
            v5 = 0
    except Exception:
        v5 = 0

    try:
        with open("data/sign_classifier.pkl", "rb") as file:
            clf = pickle.load(file)
        prediction = clf.predict([[v1, v3, v5]])[0][:2]
        sign_class_path.write_text(prediction)
    except Exception as e:
        print(f"Warning: Could not predict sign class: {e}")
        sign_class_path.write_text("0a")