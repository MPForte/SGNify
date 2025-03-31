import json
import numpy as np
import os
import re

def segment_signs(*, openpose_folder, output_path):
    # Collect keypoints and keep track of actual frame numbers
    keypoints = []
    frame_numbers = []
    
    for json_path in sorted(openpose_folder.glob("*.json")):
        # Extract frame number from filename
        filename = os.path.basename(json_path)
        match = re.search(r'^(\d+)', filename)
        if match:
            frame_num = int(match.group(1))
            frame_numbers.append(frame_num)
            
            # Load keypoints
            with json_path.open() as json_file:
                json_data = json.load(json_file)
            if json_data["people"]:
                keypoints.append(json_data["people"][0]["pose_keypoints_2d"])
            else:
                keypoints.append([0.0] * 75)
    
    # Sort frames and keypoints together
    frame_and_keypoints = sorted(zip(frame_numbers, keypoints))
    if not frame_and_keypoints:
        print("No valid frames found")
        with output_path.open("w") as json_file:
            json.dump({"start": 1, "end": 1, "t1": 1, "t2": 1, "t3": 1, "t4": 1}, json_file)
        return
    
    # Unzip the sorted pairs
    frame_numbers, keypoints = zip(*frame_and_keypoints)
    
    # Process keypoints as before
    x, y, c = np.transpose(np.reshape(keypoints, (-1, 25, 3)), (2, 1, 0))
    c = np.isclose(c, 0)
    shoulder_width = np.median(np.diff(y[[5, 2]], axis=0)[0, ~np.any(c[[5, 2]], axis=0)])
    y_elb = np.max(y[[3, 6]][:, ~np.any(c[[3, 6]], axis=0)])
    cutoff = y_elb + shoulder_width / 4

    heights = y[[4, 7]]
    heights[np.isclose(heights, 0)] = np.inf
    valid = np.any(heights < cutoff, axis=0)

    # Find indices of valid frames
    valid_indices = np.nonzero(valid)[0]
    if len(valid_indices) == 0:
        print("No valid frames detected for segmentation")
        with output_path.open("w") as json_file:
            json.dump({"start": frame_numbers[0], "end": frame_numbers[-1], 
                      "t1": frame_numbers[0], "t2": frame_numbers[0], 
                      "t3": frame_numbers[0], "t4": frame_numbers[-1]}, json_file)
        return
        
    start_idx, end_idx = valid_indices[[0, -1]]
    
    # Get actual frame numbers for start and end
    start_frame = frame_numbers[start_idx]
    end_frame = frame_numbers[end_idx]
    
    # Calculate the relative position of intermediate points
    T = end_idx - start_idx
    t1_idx = int(round(start_idx + 0.5 * T / 8))
    t2_idx = int(round(start_idx + 2.5 * T / 8))
    t3_idx = int(round(start_idx + 4.5 * T / 8))
    t4_idx = int(round(start_idx + 7 * T / 8))
    
    # Ensure indices are within bounds
    t1_idx = max(0, min(t1_idx, len(frame_numbers) - 1))
    t2_idx = max(0, min(t2_idx, len(frame_numbers) - 1))
    t3_idx = max(0, min(t3_idx, len(frame_numbers) - 1))
    t4_idx = max(0, min(t4_idx, len(frame_numbers) - 1))
    
    # Get the actual frame numbers at these indices
    output = {
        "start": start_frame,
        "end": end_frame,
        "t1": frame_numbers[t1_idx],
        "t2": frame_numbers[t2_idx],
        "t3": frame_numbers[t3_idx],
        "t4": frame_numbers[t4_idx],
    }

    with output_path.open("w") as json_file:
        json.dump(output, json_file)