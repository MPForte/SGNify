import json
import os
import pickle
import re
from os.path import join

import cv2
import mediapipe as mp
import numpy as np
import tqdm.auto as tqdm

def run_mediapipe_hands(*, output_folder, confidence, static_image_mode, keypoint_folder):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    rhand_list = []
    lhand_list = []
    with mp_hands.Hands(
        static_image_mode=static_image_mode, max_num_hands=2, min_detection_confidence=confidence
    ) as hands:
        for file in tqdm.tqdm(sorted(
            [
                f
                for f in os.listdir(f"{output_folder}/images")
                if os.path.isfile(os.path.join(f"{output_folder}/images", f))
            ]
        )):
            image = cv2.flip(cv2.imread(join(output_folder, "images", file)), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # get values
            ridx = -1
            lidx = -1
            rhand = np.zeros((1, 63))
            lhand = np.zeros((1, 63))
            rscore = 0
            lscore = 0
            if results.multi_handedness is not None:
                txt = "[%s]" % ", ".join(map(str, results.multi_handedness))
                split_string = re.split(r"\n", txt)

                if "Right" in txt:
                    ridx = split_string.index('  label: "Right"')
                    rscore = 0.7
                if "Left" in txt:
                    lidx = split_string.index('  label: "Left"')
                    lscore = 0.7

            if results.multi_hand_landmarks:
                i = 0

                image_height, image_width, _ = image.shape
                annotated_image = image.copy()

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    if len(results.multi_handedness) == 2 and ridx != -1 and lidx != -1:  # both left and right
                        if i == 0:
                            for point in mp_hands.HandLandmark:
                                normalizedLandmark = hand_landmarks.landmark[point]
                                coord_x = image_width - normalizedLandmark.x * image_width
                                coord_y = normalizedLandmark.y * image_height
                                if ridx < lidx:
                                    if normalizedLandmark is not None:
                                        rhand[0, point * 3 : (point * 3) + 2] = np.array([coord_x, coord_y])
                                        rhand[0, (point * 3) + 2] = rscore
                                else:
                                    if normalizedLandmark is not None:
                                        lhand[0, point * 3 : (point * 3) + 2] = np.array([coord_x, coord_y])
                                        lhand[0, (point * 3) + 2] = lscore
                        else:
                            for point in mp_hands.HandLandmark:
                                normalizedLandmark = hand_landmarks.landmark[point]
                                coord_x = image_width - normalizedLandmark.x * image_width
                                coord_y = normalizedLandmark.y * image_height
                                if ridx < lidx:
                                    if normalizedLandmark is not None:
                                        lhand[0, point * 3 : (point * 3) + 2] = np.array([coord_x, coord_y])
                                        lhand[0, (point * 3) + 2] = lscore
                                else:
                                    if normalizedLandmark is not None:
                                        rhand[0, point * 3 : (point * 3) + 2] = np.array([coord_x, coord_y])
                                        rhand[0, (point * 3) + 2] = rscore
                    else:
                        normalizedLandmark = []
                        for point in mp_hands.HandLandmark:
                            normalizedLandmark = hand_landmarks.landmark[point]
                            coord_x = image_width - normalizedLandmark.x * image_width
                            coord_y = normalizedLandmark.y * image_height
                            if ridx != -1:
                                if normalizedLandmark is not None:
                                    rhand[0, point * 3 : (point * 3) + 2] = np.array([coord_x, coord_y])
                                    rhand[0, (point * 3) + 2] = rscore
                            else:
                                if normalizedLandmark is not None:
                                    lhand[0, point * 3 : (point * 3) + 2] = np.array([coord_x, coord_y])
                                    lhand[0, (point * 3) + 2] = lscore

                    keypoint_fn = os.path.join(keypoint_folder, file.split(".")[0] + "_keypoints.json")
                    with open(keypoint_fn) as keypoint_file:
                        data = json.load(keypoint_file)

                    for idx, person_data in enumerate(data["people"]):
                        lhand_swap = np.zeros((1, 63))
                        rhand_swap = np.zeros((1, 63))
                        if (np.array(person_data["pose_keypoints_2d"][12:14])).any() and (
                            np.array(person_data["pose_keypoints_2d"][21:23])
                        ).any():
                            if rhand.any():
                                if np.linalg.norm(
                                    rhand[0][:2] - np.array(person_data["pose_keypoints_2d"][21:23])
                                ) < np.linalg.norm(rhand[0][:2] - np.array(person_data["pose_keypoints_2d"][12:14])):
                                    lhand_swap = rhand
                                    rhand = np.zeros((1, 63))
                            if lhand.any():
                                if np.linalg.norm(
                                    lhand[0][:2] - np.array(person_data["pose_keypoints_2d"][12:14])
                                ) < np.linalg.norm(lhand[0][:2] - np.array(person_data["pose_keypoints_2d"][21:23])):
                                    rhand_swap = lhand
                                    lhand = np.zeros((1, 63))

                        if not (np.array(person_data["hand_left_keypoints_2d"])).any():
                            if lhand_swap.any():
                                print("left hand swap")
                                person_data["hand_left_keypoints_2d"] = lhand_swap.tolist()[0]
                                lhand = np.zeros((1, 63))
                            else:
                                person_data["hand_left_keypoints_2d"] = lhand.tolist()[0]

                        if not (np.array(person_data["hand_right_keypoints_2d"])).any():
                            if rhand_swap.any():
                                person_data["hand_right_keypoints_2d"] = rhand_swap.tolist()[0]
                                rhand = np.zeros((1, 63))
                            else:
                                person_data["hand_right_keypoints_2d"] = rhand.tolist()[0]

                        if lhand.any() or lhand_swap.any():
                            lhand_list.append(file.split(".")[0])
                        if rhand.any() or rhand_swap.any():
                            rhand_list.append(file.split(".")[0])

                    cv2.imwrite(
                        os.path.join(keypoint_folder, file.split(".")[0] + "_keypoints.png"),
                        cv2.flip(annotated_image, 1),
                    )

                    keypoint_out = os.path.join(keypoint_folder, file.split(".")[0] + "_keypoints.json")
                    with open(keypoint_out, "w") as keypoint_file:
                        json.dump(data, keypoint_file)

                    i += 1

        mp_keypoints_path = os.path.join(output_folder, "mp_keypoints_{:.1f}.pkl".format(confidence))
        mp_keypoints = {"Right": rhand_list, "Left": lhand_list}
        with open(mp_keypoints_path, "wb") as f:
            pickle.dump(mp_keypoints, f, pickle.HIGHEST_PROTOCOL)
