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
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict
import copy

import cv2
import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from human_body_prior.tools.model_loader import load_vposer
import warnings
import contextlib

def fit_single_frame(img,
                     keypoints,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     sc_module,
                     result_fn='out.pkl',
                     mesh_fn='out.obj',
                     out_img_fn='overlay.png',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     scopti_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=True,
                     df_cone_height=0.5,
                     penalize_outside=True,
                     max_collisions=8,
                     point2plane=False,
                     part_segm_fn='',
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     save_meshes=True,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     ign_part_pairs=None,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     start_opt_stage=0,
                     joint_reg_path='',
                     use_joints_reg=False,
                     gen_sc_inside_weight=0.5,
                     gen_sc_outside_weight=0.0,
                     gen_sc_contact_weight=0.5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    loss_type = kwargs.get('loss_type', None)

    prev_res_path = kwargs.get('prev_res_path', None)
    if prev_res_path:
        with open(prev_res_path, 'rb') as pkl_f:
            prev_body_model = pickle.load(pkl_f)

        prev_params = {}
        for key, val in prev_body_model.items():
            if key == 'gender':
                continue
            prev_params[key] = torch.tensor(val, dtype=dtype, device=device)
        body_pose = prev_body_model.get("body_pose")
        expression = prev_body_model.get("expression")
        
        prev_pose = {
            "body": body_pose,
            "expression": expression}

        # Assumption: during isolated signs, subjects do not vary the global orientation. First frame is good.
        body_model.global_orient.requires_grad = False
        body_model.betas.requires_grad = False
        body_model.transl.requires_grad = False

        if use_hands:
            left_hand_prev_pose = torch.einsum(
                'bi,ij->bj', [torch.from_numpy(prev_body_model.get('left_hand_pose')).to(device), body_model.left_hand_components])
            right_hand_prev_pose = torch.einsum(
                'bi,ij->bj', [torch.from_numpy(prev_body_model.get('right_hand_pose')).to(device), body_model.right_hand_components])
            prev_pose.update({
                "left_hand": left_hand_prev_pose,
                "right_hand": right_hand_prev_pose
            })    
    else:
        prev_pose = None
        beta_precomputed = kwargs.get('beta_precomputed', False)
        if beta_precomputed:
            beta_path = kwargs.get('beta_path', None)
            if beta_path:
                with open(beta_path, 'rb') as pkl_f:
                    betas = (pickle.load(pkl_f)) #['shape'] # add shape if adding Yao's shape

                betas_num = body_model.betas.shape[1]
                # betas provided externally, not optimized
                body_model.betas.requires_grad = False
            else:
                print('beta_precomputed == True but no beta files (.pkl) found.')
                exit()

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

        # Upload reference handposes for SGNify
        handshape_reference = None
        if loss_type == 'sgnify':
            right_handpose_path = kwargs.get('right_handpose_path', 'None')
            if right_handpose_path != 'None':
                with open(right_handpose_path, 'rb') as pkl_f:
                    right_hand_ref = pickle.load(pkl_f).get('right_hand_pose')
                right_hand_ref_aa = torch.einsum(
                    'bi,ij->bj', [torch.from_numpy(right_hand_ref.astype(np.float32)).to(device), body_model.right_hand_components])

            left_handpose_path = kwargs.get('left_handpose_path', 'None')
            if left_handpose_path != 'None':
                with open(left_handpose_path, 'rb') as pkl_f:
                    left_hand_ref = pickle.load(pkl_f).get('left_hand_pose')
                left_hand_ref_aa = torch.einsum(
                    'bi,ij->bj', [torch.from_numpy(left_hand_ref.astype(np.float32)).to(device), body_model.left_hand_components])

                handshape_reference = {
                    "left_hand": left_hand_ref_aa,
                    "right_hand": right_hand_ref_aa
                }
            else:
                handshape_reference = {
                    "right_hand": right_hand_ref_aa
                }
                # body_model.left_hand_pose.requires_grad = False   

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        expression_precomputed = kwargs.get('expression_precomputed', False)
        if expression_precomputed:
            expression_path = kwargs.get('expression_path', None)
            if expression_path:
                with open(expression_path, 'rb') as f:
                    emoca = pickle.load(f, encoding='latin1')
                expression = emoca['exp']
                # expressions provided externally, not optimized
                body_model.expression.requires_grad = False

                # uncomment if using the jaw from SPECTRE
                jaw_pose = emoca['pose'][3:]
                body_model.jaw_pose.requires_grad = False
        else:
            if expr_weights is None:
                expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
            msg = ('Number of Body pose prior weights = {} does not match the' +
                ' number of Expression prior weights = {}')
            assert (len(expr_weights) ==
                    len(body_pose_prior_weights)), msg.format(
                        len(body_pose_prior_weights),
                        len(expr_weights))

        # indent if using jaw from SPECTRE
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)
        if prev_res_path:
            pose_embedding = torch.tensor(prev_params['pose_embedding'].cpu().numpy(), device=device, requires_grad=True) 

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                     dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    keypoint_data = torch.tensor(keypoints, dtype=dtype, device=device)

    lower_body_joints = [10,11,13,14,19,20,21,22,23,24]
    keypoint_data[:, lower_body_joints, :] = 0.0
    if keypoint_data[:, 2, :].any() and keypoint_data[:, 5, :].any():
        keypoint_data[:, 1, :] = keypoint_data[:, [2,5], :].mean(dim=1)
    if keypoint_data[:, 9, :].any() and keypoint_data[:, 12, :].any():
        keypoint_data[:, 8, :] = keypoint_data[:, [9,12], :].mean(dim=1)

    if torch.linalg.norm(keypoint_data[:, 8, :2] - keypoint_data[:, 1, :2]) < 2.5 * (torch.linalg.norm(keypoint_data[:, 0, :2] - keypoint_data[:, 1, :2])):
        keypoint_data[:, [8,9,12], 1] = keypoint_data[:, 1, 1] + 2.65 * (torch.linalg.norm(keypoint_data[:, 0, :2] - keypoint_data[:, 1, :2]))
        keypoint_data[:, 8, 2] /= 2
 
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(len(keypoints), -1)

    # compensate for missing wrist:
    H, W, _ = torch.tensor(img, dtype=dtype).shape
    if joints_conf[:, 4] == 0:
        keypoint_data[:, 4, :] = keypoint_data[:, [2,3], :].mean(dim=1)
        keypoint_data[:, 4, 1] = keypoint_data[:, 3, 1] + 0.85 * (torch.linalg.norm(keypoint_data[:, 2, :2] - keypoint_data[:, 3, :2]))
    if joints_conf[:, 7] == 0:
        keypoint_data[:, 7, :] = keypoint_data[:, [5,6], :].mean(dim=1)
        keypoint_data[:, 7, 1] = keypoint_data[:, 6, 1] + 0.85 * (torch.linalg.norm(keypoint_data[:, 5, :2] - keypoint_data[:, 6, :2]))
    ###################################################################################################

    gt_joints = keypoint_data[:, :, :2]

    if prev_res_path:
        if (torch.linalg.norm(gt_joints[:, 25, :] - gt_joints[:, 4, :]) < torch.linalg.norm(gt_joints[:, 25, :] - gt_joints[:, 7, :])):  
            joints_conf[:, 25:46]  = 0.0
        elif (torch.linalg.norm(gt_joints[:, 25, :] - gt_joints[:, 7, :]) > torch.linalg.norm(gt_joints[:, 6, :] - gt_joints[:, 7, :])):  
            joints_conf[:, 25:46]  = 0.0

        if (torch.linalg.norm(gt_joints[:, 46, :] - gt_joints[:, 7, :]) < torch.linalg.norm(gt_joints[:, 46, :] - gt_joints[:, 4, :])): 
            joints_conf[:, 46:67] = 0.0 
        elif (torch.linalg.norm(gt_joints[:, 46, :] - gt_joints[:, 4, :]) > torch.linalg.norm(gt_joints[:, 4, :] - gt_joints[:, 3, :])):  
            joints_conf[:, 46:67]  = 0.0

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        if not expression_precomputed:
            opt_weights_dict['expr_prior_weight'] = expr_weights
            # indent if using SPECTRE
            opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if interpenetration:
        opt_weights_dict['scopti_weight'] =  scopti_weights

    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # Check if all joints are visible, if not remove not visible joints
    init_joints_idxs_tmp = []
    for joint in init_joints_idxs:
        if torch.all(keypoint_data[:, joint, :].cpu() > 0):
            init_joints_idxs_tmp.append(joint)
    
    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs_tmp, device=device)

    if not prev_res_path:
        camera_translation_path = kwargs.get('camera_translation_path', None)
        if camera_translation_path:
            with open(camera_translation_path, 'rb') as pkl_f:
                init_t = torch.tensor([pickle.load(pkl_f).get('median_camera_translation')],device=device)
        else:
            edge_indices = kwargs.get('body_tri_idxs')
            init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                    use_vposer=use_vposer, vposer=vposer,
                                    pose_embedding=pose_embedding,
                                    model_type=kwargs.get('model_type', 'smpl'),
                                    focal_length=focal_length, dtype=dtype)

        camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)
        camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(handshape_reference=handshape_reference,
                               joint_weights=joint_weights,
                               prev_pose=prev_pose,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               sc_module=sc_module,
                               dtype=dtype,
                               gen_sc_inside_weight=gen_sc_inside_weight,
                               gen_sc_outside_weight=gen_sc_outside_weight,
                               gen_sc_contact_weight=gen_sc_contact_weight,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        H, W, _ = torch.tensor(img, dtype=dtype).shape

        data_weight = 1000 / H

        # Step 1: Initialization
        # Only to the first frame in a sequence
        if not prev_res_path:  # first
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=UserWarning)
                if beta_precomputed and expression_precomputed:
                    # uncomment jaw if taking it from SPECTRE
                    body_model.reset_params(body_pose=body_mean_pose,
                                                betas=torch.as_tensor(betas,device=device),
                                                expression=torch.as_tensor(expression,device=device),
                                                jaw_pose=torch.as_tensor(jaw_pose,device=device)
                                                )
                elif beta_precomputed:
                    body_model.reset_params(body_pose=body_mean_pose,
                                                betas=torch.as_tensor(betas[:betas_num],device=device),
                                                )
                elif expression_precomputed:
                    # uncomment jaw if taking it from SPECTRE
                    body_model.reset_params(body_pose=body_mean_pose,
                                                expression=torch.as_tensor(expression,device=device),
                                                jaw_pose=torch.as_tensor(jaw_pose,device=device)
                                                )                                
                else:
                    body_model.reset_params(body_pose=body_mean_pose)

            if start_opt_stage > 0:
                # Reset the parameters to mean pose
                # obtain gt joints from initialization
                if use_vposer:
                    with torch.no_grad():
                        pose_embedding.fill_(0)

                # Update the value of the translation of the camera as well as
                # the image center.
                with torch.no_grad():
                    camera.translation[:] = init_t.view_as(camera.translation)
                    camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

                # Re-enable gradient calculation for the camera translation
                camera.translation.requires_grad = True

                camera_opt_params = [
                    camera.translation, body_model.global_orient]

                camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
                    camera_opt_params,
                    **kwargs)

                # The closure passed to the optimizer
                fit_camera = monitor.create_fitting_closure(
                    camera_optimizer, body_model, camera, gt_joints,
                    camera_loss, create_graph=camera_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_full_pose=False, return_verts=False)

                camera_init_start = time.time()
                cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                        fit_camera,
                                                        camera_opt_params, body_model,
                                                        use_vposer=use_vposer,
                                                        pose_embedding=pose_embedding,
                                                        vposer=vposer)
                if camera_translation_path:
                    with open(camera_translation_path, 'rb') as pkl_f:
                        orientations = torch.tensor([pickle.load(pkl_f).get('median_global_orientation')], device=device)

            else:  # if the starting stage == 0: find global orientation first
                # The closure passed to the optimizer
                # camera_loss.reset_loss_weights({'data_weight': data_weight})
                if use_vposer:
                    with torch.no_grad():
                        pose_embedding.fill_(0)

                # If the distance between the 2D shoulders is smaller than a
                # predefined threshold then try 2 fits, the initial one and a 180
                # degree rotation
                shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                           gt_joints[:, right_shoulder_idx])
                try_both_orient = shoulder_dist.item() < side_view_thsh

                # Update the value of the translation of the camera as well as
                # the image center.
                with torch.no_grad():
                    camera.translation[:] = init_t.view_as(camera.translation)
                    camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

                # Re-enable gradient calculation for the camera translation
                camera.translation.requires_grad = True

                camera_opt_params = [camera.translation, body_model.global_orient]

                camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
                    camera_opt_params,
                    **kwargs)

                # The closure passed to the optimizer
                fit_camera = monitor.create_fitting_closure(
                    camera_optimizer, body_model, camera, gt_joints,
                    camera_loss, create_graph=camera_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_full_pose=False, return_verts=False)

                # Step 1: Optimize over the torso joints the camera translation
                # Initialize the computational graph by feeding the initial translation
                # of the camera and the initial pose of the body model.
                camera_init_start = time.time()
                cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

                # If the 2D detections/positions of the shoulder joints are too
                # close the rotate the body by 180 degrees and also fit to that
                # orientation
                if try_both_orient:
                    body_orient = body_model.global_orient.detach().cpu().numpy()
                    flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                        cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
                    flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

                    flipped_orient = torch.tensor(flipped_orient, dtype=dtype, device=device).unsqueeze(dim=0)
                    orientations = [body_orient, flipped_orient]
                else:
                    orientations = [body_model.global_orient.detach().cpu().numpy()]

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                tqdm.write('Camera initialization done after {:.4f}'.format(
                    time.time() - camera_init_start))
                tqdm.write('Camera initialization final loss {:.4f}'.format(
                    cam_init_loss_val))

        else: # from the second frame, use previous camera translation
            if left_handpose_path == 'None':
                body_model.left_hand_pose.requires_grad = False

            with open(prev_res_path, 'rb') as pkl_f:
                camera_translation = pickle.load(pkl_f).get('camera_translation')

            with torch.no_grad():
                camera.translation[:] = torch.tensor(camera_translation, device=device)
                camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

            orientations = [
                    body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation') if interactive else orientations):
            opt_start = time.time()

            if not prev_res_path:
                if start_opt_stage == 0:
                    new_params = defaultdict(global_orient=orient,
                                             body_pose=body_mean_pose,
                                             transl=body_model.transl)
                else:
                    new_params = defaultdict(global_orient=orient)
                
                if beta_precomputed:
                    new_params['betas'] = torch.tensor(betas[:betas_num], device=device)

                if expression_precomputed:
                    new_params['expression'] = torch.as_tensor(expression, device=device)
                    new_params['jaw_pose'] = torch.as_tensor(jaw_pose, device=device) 
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    body_model.reset_params(**new_params)

                if use_vposer:
                    with torch.no_grad():
                        pose_embedding.fill_(0)
            else:  # initialize with previous frame
                if expression_precomputed:
                    prev_params['expression'] = torch.as_tensor(expression, device=device)
                    prev_params['jaw_pose'] = torch.as_tensor(jaw_pose, device=device)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    body_model.reset_params(**prev_params)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights[start_opt_stage:], desc='Stage') if interactive else opt_weights[start_opt_stage:]):

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    100.7 * (curr_weights['body_pose_weight'] ** 2))
                if use_hands:
                    joint_weights[:, 25:67] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 67:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters()})
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0

            body_pose = vposer.decode(
                pose_embedding,
                output_type='aa').view(1, -1) if use_vposer else None

            model_type = kwargs.get('model_type', 'smpl')
            append_wrists = model_type == 'smpl' and use_vposer
            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                        dtype=body_pose.dtype,
                                        device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            model_output = body_model(return_verts=True, body_pose=body_pose)
            results[min_idx]['result']['body_pose'] = body_pose.detach().cpu().numpy()
            results[min_idx]['result']['pose_embedding'] = pose_embedding.detach().cpu().numpy()
            results[min_idx]['result']['gender'] = body_model.gender
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)

        if save_meshes or visualize:
            import trimesh
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()
            # Create vertex colors array (default to light pink)
            vertex_colors = np.ones((len(vertices), 4)) * [0.9, 0.5, 0.9, 1.0]

            # Create mesh with vertex colors
            out_mesh = trimesh.Trimesh(
                vertices, 
                body_model.faces, 
                vertex_colors=vertex_colors,
                process=False)
            
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            out_mesh.apply_transform(rot)

            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            if 'GPU_DEVICE_ORDINAL' in os.environ:
                os.environ['EGL_DEVICE_ID'] = os.environ['GPU_DEVICE_ORDINAL'].split(',')[0]
            import pyrender

            # Create the basic material (will be colored by vertex colors)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(1.0, 1.0, 1.0, 0.7))  # White base color to let vertex colors show

            # Create a mesh that preserves vertex colors
            mesh = pyrender.Mesh.from_trimesh(
                    out_mesh,
                    material=material,
                    smooth=False)  # Set smooth=False to preserve vertex colors

            out_mesh.export(mesh_fn)                
            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
            scene.add(mesh, 'mesh')

            # Rest of the visualization code remains the same
            height, width = img.shape[:2]
            camera_center = camera.center.detach().cpu().numpy().squeeze()
            camera_transl = camera.translation.detach().cpu().numpy().squeeze()
            camera_transl[0] *= -1.0

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl

            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length, fy=focal_length,
                cx=camera_center[0], cy=camera_center[1])
            scene.add(camera, pose=camera_pose)

            light_node = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
            scene.add(light_node, pose=camera_pose)

            r = pyrender.OffscreenRenderer(viewport_width=width,
                                            viewport_height=height,
                                            point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            img_uint8 = (img * 255).astype(np.uint8)
            img_rgba = pil_img.fromarray(img_uint8).convert('RGBA')

            # Convert rendered mesh to Image
            mesh_rgba = pil_img.fromarray(color.astype(np.uint8), 'RGBA')

            # Overlay mesh on original image
            output_img = pil_img.alpha_composite(img_rgba, mesh_rgba)

            if height > 1080:
                output_img = output_img.resize((int(width/2), int(height/2)), pil_img.ANTIALIAS)

            output_img.save(out_img_fn)

        return copy.deepcopy(out_mesh)