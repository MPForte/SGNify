import json
import pickle
from pathlib import Path
from subprocess import run

import numpy as np
import smplx
import torch
import trimesh

from utils.misc import get_end


def create_obj(pkl_in, hand, avg_pca, out_path):
    out_path.parent.mkdir(exist_ok=True)

    with open(pkl_in, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    del data["gender"]
    data["right_hand_pose"] = np.zeros((1, 45))
    data["left_hand_pose"] = np.zeros((1, 45))
    if avg_pca.shape:
        data[f"{hand}_hand_pose"] = avg_pca
    else:
        print("Warning: No data for", out_path)

    with out_path.with_suffix(".pkl").open("wb") as file:
        pickle.dump(data, file)

    dtype = torch.float32
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data = {k: torch.tensor(v, dtype=dtype, device=device) for k, v in data.items()}

    model_params = dict(
        model_path="data/models",
        create_global_orient=True,
        create_body_pose=False,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        num_expression_coeffs=50,
        num_betas=300,
        dtype=dtype,
        model_type="smplx",
        use_pca=False #,
        # num_pca_comps=45,
    )

    model = smplx.create(**model_params)
    model = model.to(device=device)

    model_output = model(**data)
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()

    out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)

    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    out_mesh.apply_transform(rot)
    out_mesh.export(out_path.with_suffix(".obj"))


def average_handpose(pkl_paths, hand, mp_frames):
    hand_pca = []
    weights = []
    for pkl_path in pkl_paths:
        with open(pkl_path, "rb") as file:
            data = pickle.load(file, encoding="latin1")
        hand_pca.append(data[f"{hand}_hand_pose"])
        weights.append(mp_frames[int(pkl_path.stem)])

    if len(hand_pca) == 0:
        return np.zeros((1, 45), dtype=np.float32)
    return np.average(hand_pca, axis=0, weights=weights)


def find_average(result_path, hand, mp_frames, sign_class, segment, out_prefix=""):
    pkl_path = result_path.joinpath(hand, "results")
    pkl_files = sorted([n for n in pkl_path.glob("*.pkl")])
    pkl_file_first = "data/EMPTY.pkl"

    if (
        sign_class[1] == "a" or sign_class != "1b" and hand == "left"
    ) and sign_class != "-1":  # only one for the entire sequence
        create_obj(
            pkl_in=pkl_file_first,
            hand=hand,
            avg_pca=average_handpose(
                [path for path in pkl_files if segment["t1"] <= int(path.stem) <= segment["t4"]], hand, mp_frames
            ),
            out_path=result_path.joinpath(out_prefix, f"ref_{hand}"),
        )
    elif sign_class[1] == "b" and hand == "right" or sign_class == "1b" and hand == "left" or sign_class == "-1":
        # reference 1
        create_obj(
            pkl_in=pkl_file_first,
            hand=hand,
            avg_pca=average_handpose(
                [
                    path
                    for path in pkl_files
                    if segment["t1"] <= int(path.stem) <= round(segment["t2"] + (segment["t3"] - segment["t2"]) / 2)
                ],
                hand,
                mp_frames,
            ),
            out_path=result_path.joinpath(out_prefix, f"ref_{hand}_1"),
        )

        # reference 2
        create_obj(
            pkl_in=pkl_file_first,
            hand=hand,
            avg_pca=average_handpose(
                [
                    path
                    for path in pkl_files
                    if round(segment["t2"] + (segment["t3"] - segment["t2"]) / 2) <= int(path.stem) <= segment["t4"]
                ],
                hand,
                mp_frames,
            ),
            out_path=result_path.joinpath(out_prefix, f"ref_{hand}_2"),
        )


def find_rps(result_path, segment_path, sign_class="-1", out_prefix=""):
    result_path = Path(result_path)
    segment_path = Path(segment_path)
    sign_class = sign_class
    out_prefix = out_prefix

    # frames with MP keypoints
    mp_keypoints_path = result_path.joinpath("mp_keypoints_weight.pkl")
    with mp_keypoints_path.open("rb") as file:
        mp_frames = pickle.load(file)

    with segment_path.open() as json_file:
        segment = json.load(json_file)

    find_average(
        result_path=result_path.joinpath("rps"),
        hand="right",
        mp_frames=mp_frames["Right"],
        sign_class=sign_class,
        out_prefix=out_prefix,
        segment=segment,
    )

    if sign_class[0] != "0":
        find_average(
            result_path=result_path.joinpath("rps"),
            hand="left",
            mp_frames=mp_frames["Left"],
            sign_class=sign_class,
            out_prefix=out_prefix,
            segment=segment,
        )


def compute_rps(*, sign_class, rps_folder, result_folder, right_interp_folder, left_interp_folder, segment_path):
    find_rps(result_path=result_folder, sign_class=sign_class, segment_path=segment_path)

    with segment_path.open() as json_file:
        segment = json.load(json_file)

    # Right RPS
    if sign_class[1] == "b":
        print("Interpolating right hand...")
        interpolate(
            pkl_1=rps_folder.joinpath("ref_right_1.pkl"),
            pkl_2=rps_folder.joinpath("ref_right_2.pkl"),
            save_folder=right_interp_folder,
            reference_start=segment["start"],
            reference_end=segment["end"],
            end_frame=get_end(result_folder),
        )

    # Left RPS
    if sign_class == "1b":
        print("Interpolating left hand...")
        interpolate(
            pkl_1=rps_folder.joinpath("ref_left_1.pkl"),
            pkl_2=rps_folder.joinpath("ref_left_2.pkl"),
            save_folder=left_interp_folder,
            reference_start=segment["start"],
            reference_end=segment["end"],
            end_frame=get_end(result_folder),
        )


def interpolate(*, pkl_1, pkl_2, save_folder, reference_start, reference_end, end_frame):
    return run(
        [
            "python",
            "utils/interpolate.py",
            "--pkl",
            pkl_1,
            pkl_2,
            "--save_folder",
            save_folder,
            "--config",
            "SGNify/cfg_files/fit_smplifyx_ref_sv.yaml",
            "--reference_start",
            str(reference_start),
            "--reference_end",
            str(reference_end),
            "--end_frame",
            str(end_frame),
        ],
        check=True,
    )


def compute_valid_frames(result_folder, segment):
    # frames with MP keypoints
    with result_folder.joinpath("mp_keypoints_weight.pkl").open("rb") as file:
        mp_frames = pickle.load(file)

    reconstruct_left = []
    for i in range(1, (mp_frames["Left"]).size):
        if mp_frames["Left"][i] > 0 and segment["t1"] <= i <= segment["t4"]:
            reconstruct_left.append(i)
    reconstruct_left = sorted(reconstruct_left)

    reconstruct_right = []
    for i in range(1, (mp_frames["Right"]).size):
        if mp_frames["Right"][i] > 0 and segment["t1"] <= i <= segment["t4"]:
            reconstruct_right.append(i)
    reconstruct_right = sorted(reconstruct_right)

    return [reconstruct_left, reconstruct_right]