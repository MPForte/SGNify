# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import argparse
import os
import pickle
import shutil

import numpy as np
import pyrender
import rotation_conversion
import smplx
import torch
import trimesh
from cmd_parser import parse_config
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def key_positions(pkl_paths, model, hand, reference_start, reference_end):
    if hand == "left":
        components = model.left_hand_components
    else:
        components = model.right_hand_components

    # get quaternion start
    with open(pkl_paths[0], "rb") as f:
        data_start = pickle.load(f, encoding="latin1")
    hand_start = torch.einsum(
        "bi,ij->bj", [torch.tensor(data_start.get(hand + "_hand_pose"), device=device, dtype=dtype), components]
    )
    hand_start_quat = rotation_conversion.axis_angle_to_quaternion(torch.reshape(hand_start, (15, 3)))

    # get quaternion end
    with open(pkl_paths[1], "rb") as f:
        data_end = pickle.load(f, encoding="latin1")
    hand_end = torch.einsum(
        "bi,ij->bj", [torch.tensor(data_end.get(hand + "_hand_pose"), device=device, dtype=dtype), components]
    )
    hand_end_quat = rotation_conversion.axis_angle_to_quaternion(torch.reshape(hand_end, (15, 3)))

    # interpolate
    key_times = [0, reference_end - reference_start]
    times = np.linspace(0, reference_end - reference_start, reference_end - reference_start)
    interp = []
    for j in range(0, 15):
        key_rots = R.from_quat([hand_start_quat[j].detach().cpu().numpy(), hand_end_quat[j].detach().cpu().numpy()])
        slerp = Slerp(key_times, key_rots)
        interp.append(slerp(times).as_quat())

    hand_pose_interp = np.zeros((reference_end - reference_start, 15, 4))
    for step in range(0, reference_end - reference_start):
        for joint in range(0, 15):
            hand_pose_interp[step, joint, :] = interp[joint][step]

    return hand_pose_interp


def quat2pca(hand_pose_interp, model, hand):
    if hand == "left":
        components = model.left_hand_components
    else:
        components = model.right_hand_components

    hand_pose_aa = rotation_conversion.quaternion_to_axis_angle(
        torch.tensor(hand_pose_interp, dtype=dtype, device=device)
    )
    inverse = np.linalg.pinv(components.detach().cpu().numpy())
    hand_pose_t = torch.einsum(
        "bi,ij->bj", [torch.reshape(hand_pose_aa, (1, 45)), torch.from_numpy(inverse).to(device)]
    )

    return hand_pose_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", nargs="+", type=str, required=True, help="The pkl files that will be read")
    parser.add_argument("--save_folder", type=str, required=False, help="The folder where to store the pkl files")
    parser.add_argument("--reference_start", type=int, required=False, help="The first frame of the RPS")
    parser.add_argument("--reference_end", type=int, required=False, help="The final frame of the RPS")
    parser.add_argument("--end_frame", type=int, required=False, help="The final frame")

    args, remaining = parser.parse_known_args()
    pkl = args.pkl
    save_folder = args.save_folder
    reference_start = args.reference_start
    reference_end = args.reference_end
    end_frame = args.end_frame

    args = parse_config(remaining)

    dtype = torch.float32
    use_cuda = args.get("use_cuda", True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_type = args.get("model_type", "smpl")
    print("Model type:", model_type)
    print(args.get("model_folder"))
    model_params = dict(
        model_path=args.get("model_folder"),
        #  joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=not args.get("use_vposer"),
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
        **args
    )

    model = smplx.create(**model_params)
    model = model.to(device=device)

    batch_size = args.get("batch_size", 1)

    # create poses
    if "left" in save_folder:
        hand = "left"
        left_hand_pose_interp = key_positions(pkl, model, "left", reference_start, reference_end)
    elif "right" in save_folder:
        hand = "right"
        right_hand_pose_interp = key_positions(pkl, model, "right", reference_start, reference_end)
    else:
        left_hand_pose_interp = key_positions(pkl, model, "left", reference_start, reference_end)
        right_hand_pose_interp = key_positions(pkl, model, "right", reference_start, reference_end)

    for t in range(0, reference_end - reference_start):
        with open(pkl[0], "rb") as f:
            data = pickle.load(f, encoding="latin1")

        est_params = {}
        for key, val in data.items():
            if key == "gender" or key == "left_hand_pose" or key == "right_hand_pose":
                continue
            est_params[key] = torch.tensor(val, dtype=dtype, device=device)

        # go from quat to PCA
        if hand == "left":
            left_hand_pose_t = quat2pca(left_hand_pose_interp[t], model, "left")
            results = {"left_hand_pose": left_hand_pose_t.detach().cpu().numpy()}
        else:
            right_hand_pose_t = quat2pca(right_hand_pose_interp[t], model, "right")
            results = {"right_hand_pose": right_hand_pose_t.detach().cpu().numpy()}
        with open(os.path.join(save_folder, "{:03d}.pkl".format(t + reference_start)), "wb") as result_file:
            pickle.dump(results, result_file)

        if hand == "left":
            model_output = model(left_hand_pose=left_hand_pose_t, **est_params)
        else:
            model_output = model(right_hand_pose=right_hand_pose_t, **est_params)
        vertices = model_output.vertices.detach().cpu().numpy().squeeze()

        out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(0.9, 0.5, 0.9, 1)
        )

        mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh.export(os.path.join(save_folder, "{:03d}.obj".format(t + reference_start)))

    # Save all before_reference_start = reference_start and after_reference_end = reference_end
    if reference_start > 0:
        for before in range(0, reference_start):
            shutil.copy(
                os.path.join(save_folder, "{:03d}.pkl".format(reference_start)),
                os.path.join(save_folder, "{:03d}.pkl".format(before)),
            )

    if reference_end < end_frame:
        for after in range(reference_end, end_frame + 1):
            shutil.copy(
                os.path.join(save_folder, "{:03d}.pkl".format(reference_end - 1)),
                os.path.join(save_folder, "{:03d}.pkl".format(after)),
            )
