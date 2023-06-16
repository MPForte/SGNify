# -*- coding: utf-8 -*-
# modified from https://github.com/filby89/spectre

import argparse
import collections
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.transform import estimate_transform, warp
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa
from spectre.config import cfg as spectre_cfg
from spectre.datasets.data_utils import landmarks_interpolate
from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.face_tracker import (
    FaceTracker,
)
from spectre.external.Visual_Speech_Recognition_for_Multiple_Languages.tracker.utils import (
    get_landmarks,
)
from spectre.src.spectre import SPECTRE
from spectre.src.utils.util import tensor2video


def extract_frames(image_paths):
    face_tracker = FaceTracker()

    face_info = collections.defaultdict(list)

    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)

        detected_faces = face_tracker.face_detector(image, rgb=False)
        landmarks, scores = face_tracker.landmark_detector(image, detected_faces, rgb=False)
        face_info["bbox"].append(detected_faces)
        face_info["landmarks"].append(landmarks)
        face_info["landmarks_scores"].append(scores)

    return get_landmarks(face_info)


def crop_face(frame, landmarks, scale=1.0):
    image_size = 224
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    src_pts = np.array(
        [
            [center[0] - size / 2, center[1] - size / 2],
            [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2],
        ]
    )
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform("similarity", src_pts, DST_PTS)

    return tform


def main(args):
    args.crop_face = True
    spectre_cfg.pretrained_modelpath = "spectre/pretrained/spectre_model.tar"
    spectre_cfg.model.use_tex = False

    spectre = SPECTRE(spectre_cfg, args.device)
    spectre.eval()

    image_paths = [str(n) for n in sorted(Path(args.images_folder).glob("*"))]

    # PAOLA: TODO we could avoid to extract the frames again
    landmarks = extract_frames(image_paths)
    if args.crop_face:
        landmarks = landmarks_interpolate(landmarks)
        if landmarks is None:
            print("No faces detected in input {}".format(args.input))

    original_video_length = len(image_paths)
    """ SPECTRE uses a temporal convolution of size 5.
    Thus, in order to predict the parameters for a contiguous video with need to
    process the video in chunks of overlap 2, dropping values which were computed from the
    temporal kernel which uses pad 'same'. For the start and end of the video we
    pad using the first and last frame of the video.
    e.g., consider a video of size 48 frames and we want to predict it in chunks of 20 frames
    (due to memory limitations). We first pad the video two frames at the start and end using
    the first and last frames correspondingly, making the video 52 frames length.

    Then we process independently the following chunks:
    [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
     [16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]
     [32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51]]

     In the first chunk, after computing the 3DMM params we drop 0,1 and 18,19, since they were computed
     from the temporal kernel with padding (we followed the same procedure in training and computed loss
     only from valid outputs of the temporal kernel) In the second chunk, we drop 16,17 and 34,35, and in
     the last chunk we drop 32,33 and 50,51. As a result we get:
     [2..17], [18..33], [34..49] (end included) which correspond to all frames of the original video
     (removing the initial padding).
    """

    # pad
    image_paths.insert(0, image_paths[0])
    image_paths.insert(0, image_paths[0])
    image_paths.append(image_paths[-1])
    image_paths.append(image_paths[-1])

    landmarks.insert(0, landmarks[0])
    landmarks.insert(0, landmarks[0])
    landmarks.append(landmarks[-1])
    landmarks.append(landmarks[-1])

    landmarks = np.array(landmarks)

    L = 50  # chunk size

    # create lists of overlapping indices
    indices = list(range(len(image_paths)))
    overlapping_indices = [indices[i : i + L] for i in range(0, len(indices), L - 4)]

    if len(overlapping_indices[-1]) < 5:
        # if the last chunk has less than 5 frames, pad it with the semilast frame
        overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
        overlapping_indices[-2] = np.unique(overlapping_indices[-2]).tolist()
        overlapping_indices = overlapping_indices[:-1]

    overlapping_indices = np.array(overlapping_indices)

    image_paths = np.array(image_paths)  # do this to index with multiple indices
    all_shape_images = []
    all_images = []
    all_poses = []
    all_expressions = []
    all_shapes = []

    with torch.no_grad():
        for chunk_id in range(len(overlapping_indices)):
            print(
                "Processing frames {} to {}".format(overlapping_indices[chunk_id][0], overlapping_indices[chunk_id][-1])
            )
            image_paths_chunk = image_paths[overlapping_indices[chunk_id]]

            landmarks_chunk = landmarks[overlapping_indices[chunk_id]] if args.crop_face else None

            images_list = []

            """ load each image and crop it around the face if necessary """
            for j in range(len(image_paths_chunk)):
                frame = cv2.imread(image_paths_chunk[j])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                kpt = landmarks_chunk[j]

                tform = crop_face(frame, kpt, scale=1.6)
                cropped_image = warp(frame, tform.inverse, output_shape=(224, 224))

                images_list.append(cropped_image.transpose(2, 0, 1))

            images_array = (
                torch.from_numpy(np.array(images_list)).type(dtype=torch.float32).to(args.device)
            )  # K,224,224,3

            codedict, initial_deca_exp, initial_deca_jaw = spectre.encode(images_array)
            codedict["exp"] = codedict["exp"] + initial_deca_exp
            codedict["pose"][..., 3:] = codedict["pose"][..., 3:] + initial_deca_jaw

            for key in codedict.keys():
                """filter out invalid indices - see explanation at the top of the function"""

                if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                    pass
                elif chunk_id == 0:
                    codedict[key] = codedict[key][:-2]
                elif chunk_id == len(overlapping_indices) - 1:
                    codedict[key] = codedict[key][2:]
                else:
                    codedict[key] = codedict[key][2:-2]
            opdict, visdict = spectre.decode(codedict, rendering=True, vis_lmk=False, return_vis=True)
            all_shape_images.append(visdict["shape_images"].detach().cpu())
            all_images.append(codedict["images"].detach().cpu())
            # PAOLA: needed for SGNify
            all_poses.append(codedict["pose"].detach().cpu())
            all_expressions.append(codedict["exp"].detach().cpu())
            all_shapes.append(codedict["shape"].detach().cpu())

    # PAOLA: needed for SGNify
    poses = (torch.cat(all_poses, dim=0))[2:-2]  # remove padding
    expressions = (torch.cat(all_expressions, dim=0))[2:-2]  # remove padding
    shapes = (torch.cat(all_shapes, dim=0))[2:-2]  # remove padding

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    for frame in range(poses.size()[0]):
        codedict_frame = {"exp": expressions[frame], "pose": poses[frame], "shape": shapes[frame]}
        with open(f"{args.output_folder}/spectre_{frame+1}.pkl", "wb") as f:
            pickle.dump(codedict_frame, f, protocol=pickle.HIGHEST_PROTOCOL)

    vid_shape = tensor2video(torch.cat(all_shape_images, dim=0))[2:-2]  # remove padding
    vid_orig = tensor2video(torch.cat(all_images, dim=0))[2:-2]  # remove padding
    grid_vid = np.concatenate((vid_shape, vid_orig), axis=2)

    assert original_video_length == len(vid_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DECA: Detailed Expression Capture and Animation")

    parser.add_argument("--images_folder", type=str)
    parser.add_argument(
        "--output_folder",
        default="examples/results",
        type=str,
        help="path to the output directory, where results(obj, txt files) will be stored.",
    )
    parser.add_argument("--device", default="cuda", type=str, help="set device, cpu for using cpu")
    parser.add_argument("--fps", default=24, type=int)

    main(parser.parse_args())
