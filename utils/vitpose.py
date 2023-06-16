import json

import numpy as np
import tqdm.auto as tqdm
from PIL import Image
from ultralytics import YOLO
from vitpose.utils.inference import vitpose_inference_model
from vitpose.utils.visualization import draw_points_and_skeleton, joints_dict


def run_vitpose(*, images_folder, output_folder):
    det_model = YOLO("data/yolov8n.pt")
    pose_model = vitpose_inference_model(batch_size=1, model_mode="huge", weights_dir="data")

    for image_path in tqdm.tqdm(sorted(images_folder.glob("*"))):
        im = Image.open(image_path)
        x1, y1, x2, y2 = det_model(im, classes=0, max_det=1, verbose=False)[0].boxes.xyxy[0].tolist()
        pose_preds = pose_model([np.array(im.crop((x1, y1, x2, y2)))])[0]
        pose_preds[:, 0] += y1
        pose_preds[:, 1] += x1

        Image.fromarray(
            draw_points_and_skeleton(
                np.array(im).copy(),
                pose_preds,
                joints_dict()["coco"]["skeleton"],
                person_index=0,
                points_color_palette="gist_rainbow",
                skeleton_color_palette="jet",
                points_palette_samples=10,
                confidence_threshold=0.4,
            )
        ).save(output_folder.joinpath(image_path.stem + "_rendered.png"))

        pose_preds[:, [0, 1]] = pose_preds[:, [1, 0]]

        vitpose_op = np.zeros([25, 3])
        vitpose_op[[0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]] = pose_preds
        vitpose_op[1] = np.mean(pose_preds[[5, 6]], axis=0)
        vitpose_op[1, 2] = np.min(pose_preds[[5, 6], 2])

        vitpose_op[8] = np.mean(pose_preds[[11, 12]], axis=0)
        vitpose_op[8, 2] = np.min(pose_preds[[11, 12], 2])

        vitpose_op[vitpose_op[:, 2] < 0.3] = 0.0

        with output_folder.joinpath(image_path.stem + "_keypoints.json").open("w") as json_file:
            json.dump(
                {
                    "version": 1.3,
                    "people": [
                        {
                            "person_id": [-1],
                            "pose_keypoints_2d": vitpose_op.flatten().tolist(),
                            "face_keypoints_2d": [0.0] * 210,
                            "hand_left_keypoints_2d": [0.0] * 63,
                            "hand_right_keypoints_2d": [0.0] * 63,
                            "pose_keypoints_3d": [],
                            "face_keypoints_3d": [],
                            "hand_left_keypoints_3d": [],
                            "hand_right_keypoints_3d": [],
                        }
                    ],
                },
                json_file,
            )
