import json

import numpy as np


def segment_signs(*, openpose_folder, output_path):
    keypoints = []
    for json_path in sorted(openpose_folder.glob("*.json")):
        with json_path.open() as json_file:
            json_data = json.load(json_file)
        if json_data["people"]:
            keypoints.append(json_data["people"][0]["pose_keypoints_2d"])
        else:
            keypoints.append([0.0] * 75)

    x, y, c = np.transpose(np.reshape(keypoints, (-1, 25, 3)), (2, 1, 0))
    c = np.isclose(c, 0)
    shoulder_width = np.median(np.diff(y[[5, 2]], axis=0)[0, ~np.any(c[[5, 2]], axis=0)])
    y_elb = np.max(y[[3, 6]][:, ~np.any(c[[3, 6]], axis=0)])
    cutoff = y_elb + shoulder_width / 4

    heights = y[[4, 7]]
    heights[np.isclose(heights, 0)] = np.inf
    valid = np.any(heights < cutoff, axis=0)

    start, end = np.nonzero(valid)[0][[0, -1]]

    T = end - start
    output = {
        "start": int(round(start)) + 1,
        "end": int(round(end)) + 1,
        "t1": int(round(start + 0.5 * T / 8)) + 1,
        "t2": int(round(start + 2.5 * T / 8)) + 1,
        "t3": int(round(start + 4.5 * T / 8)) + 1,
        "t4": int(round(start + 7 * T / 8)) + 1,
    }

    with output_path.open("w") as json_file:
        json.dump(output, json_file)
