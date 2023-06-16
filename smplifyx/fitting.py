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

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn

from mesh_viewer import MeshViewer
import utils
from selfcontact.losses import SelfContactLoss


@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               vposer=None,
               use_vposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    body_pose = vposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
    if use_vposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(loss_type, **kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    elif loss_type == 'temp_smplify':
        return TemporalSMPLifyLoss(loss_type, **kwargs)
    elif loss_type == 'sgnify':
        return SGNifyLoss(loss_type, **kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


def L2Loss(pose_curr, pose_ref, size, smooth_weight):
    quat_pose_curr = axis_angle_to_rotation_6d(torch.reshape(pose_curr, size))
    quat_pose_ref = axis_angle_to_rotation_6d(torch.reshape(pose_ref, size))
    loss = (torch.sum((quat_pose_curr - quat_pose_ref) ** 2) * smooth_weight ** 2)
    return loss


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    # https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    matrix = o.reshape(quaternions.shape[:-1] + (3, 3))

    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


class SMPLifyLoss(nn.Module):

    def __init__(self,
                 loss_type=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 reduction='sum',
                 sc_module=None,
                 gen_sc_inside_weight=0.5,
                 gen_sc_outside_weight=0.0,
                 gen_sc_contact_weight=0.5,
                 scopti_weight=0.0,
                 beta_precomputed=True,
                 expression_precomputed=True,
                 prev_body_pose=None,
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        if self.interpenetration:
            self.sc_crit = SelfContactLoss(
                contact_module=sc_module,
                inside_loss_weight=gen_sc_inside_weight,
                outside_loss_weight=gen_sc_outside_weight,
                contact_loss_weight=gen_sc_contact_weight,
                align_faces=True,
                use_hd=True,
                test_segments=True,
                device='cuda',
                model_type='smplx'
            )

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('beta_precomputed',
                             torch.tensor(beta_precomputed, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expression_precomputed',
                                 torch.tensor(expression_precomputed, dtype=dtype))
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(kwargs.get('coll_loss_weight'), dtype=dtype))

        self.body_boneorientationpoint1 = [int(e.split(',')[0]) for e in kwargs[
            'body_boneorientation_pairs']]
        self.body_boneorientationpoint2 = [int(e.split(',')[1]) for e in kwargs[
            'body_boneorientation_pairs']]

        self.hand_boneorientationpoint1_base = [int(e.split(',')[0]) for e in kwargs[
            'hand_boneorientation_pairs']]
        self.hand_boneorientationpoint2_base = [int(e.split(',')[1]) for e in kwargs[
            'hand_boneorientation_pairs']]

        lhand_start = 25
        rhand_start = 46
        self.lhand_boneorientationpoint1 = [
            e + lhand_start for e in self.hand_boneorientationpoint1_base]
        self.lhand_boneorientationpoint2 = [
            e + lhand_start for e in self.hand_boneorientationpoint2_base]

        self.rhand_boneorientationpoint1 = [
            e + rhand_start for e in self.hand_boneorientationpoint1_base]
        self.rhand_boneorientationpoint2 = [
            e + rhand_start for e in self.hand_boneorientationpoint2_base]

        self.boneorientation_loss = kwargs.get('boneorientation_loss', False)

        self.left_handpose_path = kwargs.get('left_handpose_path', 'None')

        self.loss_type = loss_type

        if self.interpenetration:
            self.register_buffer('scopti_weight', torch.tensor(scopti_weight, dtype=dtype))

        self.prev_res_path = kwargs.get('prev_res_path')
        if self.prev_res_path or self.loss_type != 'sgnify':
            standing_weight = 50
        else:
            standing_weight = 10000
        self.register_buffer('standing_weight', torch.tensor(standing_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                **kwargs):
        projected_joints = camera(body_model_output.joints)
        # Calculate the weights for each joints
        weights = (joint_weights * joints_conf
                   if self.use_joints_conf else
                   joint_weights).unsqueeze(dim=-1)

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)

        boneorientation_loss = 0.0
        if self.boneorientation_loss:
            boneorientation_diff_body = self.robustifier(
                (gt_joints[:, self.body_boneorientationpoint1, :] - gt_joints[:, self.body_boneorientationpoint2, :]) -
                (projected_joints[:, self.body_boneorientationpoint1, :] - projected_joints[:, self.body_boneorientationpoint2, :]))
            boneorientation_loss += torch.sum(
                boneorientation_diff_body) * self.data_weight ** 2
            if torch.any(joints_conf[:, 46:67] > 0):
                boneorientation_diff_rhand = self.robustifier(
                    (gt_joints[:, self.rhand_boneorientationpoint1, :] - gt_joints[:, self.rhand_boneorientationpoint2, :]) -
                    (projected_joints[:, self.rhand_boneorientationpoint1, :] - projected_joints[:, self.rhand_boneorientationpoint2, :]))
                boneorientation_loss += torch.sum(
                    boneorientation_diff_rhand) * self.data_weight ** 2
            if torch.any(joints_conf[:, 25:46] > 0) and self.left_handpose_path != 'None':
                boneorientation_diff_lhand = self.robustifier(
                    (gt_joints[:, self.lhand_boneorientationpoint1, :] - gt_joints[:, self.lhand_boneorientationpoint2, :]) -
                    (projected_joints[:, self.lhand_boneorientationpoint1, :] - projected_joints[:, self.lhand_boneorientationpoint2, :]))
                boneorientation_loss += torch.sum(
                    boneorientation_diff_lhand) * self.data_weight ** 2

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = 0.0
        if self.beta_precomputed == False:
            shape_loss = torch.sum(self.shape_prior(
                body_model_output.betas)) * self.shape_weight ** 2

        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        standing_loss = 0.0
        # If the lower body keypoints are not visible assume that the legs are straight
        lower_body_joints = [11,22,23,24,14,19,20,21]
        valid_lower_body_joints = []
        for joint in lower_body_joints:
            if joints_conf[:, joint] != 0:
                valid_lower_body_joints.append(joint) 
        if not valid_lower_body_joints:
            stand_body_pose = body_pose.detach().clone() 
            # 12-14 L_Knee, 15-17 R_Knee, 9-11, 12-14
            # 0-2 Pelvis - not here
            # 3-5 L_Hip - 0-2
            # 6-8 R_Hip - 3-5
            # 9-11 Spine_01, 18-20 Spine_02, 27-29 Spine_03, 6-8, 15-17, 24-26
            if self.standing_weight > 50:
                stand_body_pose_idx = np.arange(0,45,1)
            else:
                stand_body_pose_idx = np.arange(0,33,1)
            stand_body_pose[:, stand_body_pose_idx] = 0.0
            standing_loss = L2Loss(body_pose,
                                stand_body_pose,
                                (21, 3), self.standing_weight)

            no_rot = body_model_output.full_pose[:,0:3].detach().clone() 
            
            no_rot[:,0] = np.pi
            if self.standing_weight > 50:
                no_rot[:,1] = 0
                no_rot[:,2] = 0
            standing_loss += L2Loss(body_model_output.full_pose[:,0:3],
                                    no_rot,
                                    (1, 3), self.standing_weight ** 2)

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            if self.expression_precomputed == False:
                expression_loss = torch.sum(self.expr_prior(
                    body_model_output.expression)) * \
                    self.expr_prior_weight ** 2

                # indent when using SPECTRE
                if hasattr(self, 'jaw_prior'):
                    jaw_prior_loss = torch.sum(
                        self.jaw_prior(
                            body_model_output.jaw_pose.mul(
                                self.jaw_prior_weight)))

        # ==== general self contact loss ====
        faces_angle_loss, gsc_contact_loss = 0.0, 0.0
        if self.interpenetration and self.scopti_weight > 0:
            gsc_contact_loss, faces_angle_loss = \
                self.sc_crit(vertices=body_model_output.vertices)
            gsc_contact_loss = self.scopti_weight * gsc_contact_loss
            faces_angle_loss = 0.1 * faces_angle_loss

        # COMMENT SHAPE LOSS
        total_loss = (joint_loss + pprior_loss + boneorientation_loss + shape_loss +
                      angle_prior_loss + gsc_contact_loss + faces_angle_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss + standing_loss)
        return total_loss


class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))
        return depth_loss + joint_loss


class TemporalSMPLifyLoss(SMPLifyLoss):
    def __init__(self,
                 loss_type,
                 prev_pose,
                 **kwargs):

        super(TemporalSMPLifyLoss, self).__init__(**kwargs)

        if prev_pose is not None:
            self.register_buffer(
                'body_prev_pose', torch.tensor(prev_pose["body"], dtype=torch.float32))
            self.register_buffer(
                'expression_prev_pose', torch.tensor(prev_pose["expression"], dtype=torch.float32))
            if self.use_hands:
                self.register_buffer(
                    'left_hand_prev_pose', prev_pose["left_hand"].clone().detach().requires_grad_(True))
                self.register_buffer(
                    'right_hand_prev_pose', prev_pose["right_hand"].clone().detach().requires_grad_(True))

        self.body_temp_smooth_weight = kwargs.get('body_temp_smooth_weight')

        if self.use_hands:
            self.hand_temp_smooth_weight = kwargs.get(
                'hand_temp_smooth_weight')

        self.loss_type = loss_type

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                **kwargs):

        frame_wise_loss = super(TemporalSMPLifyLoss, self).forward(body_model_output,
                                                                   camera=camera,
                                                                   gt_joints=gt_joints,
                                                                   body_model_faces=body_model_faces,
                                                                   joints_conf=joints_conf,
                                                                   joint_weights=joint_weights,
                                                                   pose_embedding=pose_embedding,
                                                                   use_vposer=use_vposer,
                                                                   **kwargs)

        temporal_loss = 0.0
        if hasattr(self, 'body_prev_pose') and torch.any((body_model_output.full_pose[:, 3:66]).detach().cpu() > 0):
            body_temp_smooth_weight = self.body_temp_smooth_weight
            body_loss = L2Loss(body_model_output.full_pose[:, 3:66],
                               self.body_prev_pose,
                               (21, 3), body_temp_smooth_weight)

            expression_loss = 0.0
            if self.use_face and self.expression_precomputed == False:
                expression_loss = torch.sum((body_model_output.expression -
                                             self.expression_prev_pose) ** 2) * 10

            right_hand_loss = 0.0
            right_wrist_loss = 0.0
            if self.use_hands and torch.any((body_model_output.right_hand_pose).detach().cpu() > 0):
                right_hand_loss = L2Loss(body_model_output.right_hand_pose,
                                         self.right_hand_prev_pose,
                                         (15, 3), self.hand_temp_smooth_weight)

                if torch.median(joints_conf[:, 46:67]) == 0.0:
                    right_wrist_loss = L2Loss(body_model_output.full_pose[:, 63:66],
                                    self.body_prev_pose[:, 60:63],
                                    (1, 3), 100)

            left_hand_loss = 0.0
            left_wrist_loss = 0.0
            if self.use_hands and torch.any((body_model_output.left_hand_pose).detach().cpu() > 0):
                left_hand_loss = L2Loss(body_model_output.left_hand_pose,
                                        self.left_hand_prev_pose,
                                        (15, 3), self.hand_temp_smooth_weight)

                if torch.median(joints_conf[:, 25:46]) == 0.0:
                    left_wrist_loss = L2Loss(body_model_output.full_pose[:, 60:63],
                                    self.body_prev_pose[:, 57:60],
                                    (1, 3), 100)
            temporal_loss = body_loss + right_hand_loss + left_hand_loss + \
                right_wrist_loss + left_wrist_loss + expression_loss # + arms_loss

        total_loss = frame_wise_loss + temporal_loss

        return total_loss


class SGNifyLoss(TemporalSMPLifyLoss):
    def __init__(self,
                 loss_type,
                 handshape_reference,
                 **kwargs):
        super(SGNifyLoss, self).__init__(loss_type, **kwargs)

        self.use_symmetry = kwargs.get('use_symmetry')
        self.symmetry_weight = kwargs.get('symmetry_weight')

        if handshape_reference:
            if "right_hand" in handshape_reference:
                self.register_buffer('right_hand_ref',
                                     handshape_reference["right_hand"].clone().detach().requires_grad_(True))
            if "left_hand" in handshape_reference:
                self.register_buffer('left_hand_ref',
                                     handshape_reference["left_hand"].clone().detach().requires_grad_(True))
            else:
                self.register_buffer('left_hand_ref', None)
        else:
            self.register_buffer('right_hand_ref', None)
            self.register_buffer('left_hand_ref', None)

        self.right_reference_weight = kwargs.get('right_reference_weight')
        self.left_reference_weight = kwargs.get('left_reference_weight')

        self.loss_type = loss_type

    def forward(self, body_model_output, camera, gt_joints, joints_conf,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                **kwargs):

        temporal_loss = super(SGNifyLoss, self).forward(body_model_output,
                                                        camera=camera,
                                                        gt_joints=gt_joints,
                                                        body_model_faces=body_model_faces,
                                                        joints_conf=joints_conf,
                                                        joint_weights=joint_weights,
                                                        pose_embedding=pose_embedding,
                                                        use_vposer=use_vposer,
                                                        **kwargs)

        symmetry_loss = 0.0
        if torch.any((body_model_output.right_hand_pose).detach().cpu() > 0) and torch.any((body_model_output.left_hand_pose).detach().cpu() > 0):
            # Calculate the loss due to hand symmetry
            if self.use_symmetry:
                flip_vector = torch.tensor([1.0, -1.0, -1.0], device='cuda')
                flipped_left_hand_pose = torch.reshape((torch.reshape(
                    body_model_output.left_hand_pose, (15, 3)) * flip_vector), (-1,))
                symmetry_loss = L2Loss(body_model_output.right_hand_pose,
                                       flipped_left_hand_pose,
                                       (15, 3), self.symmetry_weight)

        right_handpose_loss = 0.0
        if self.right_hand_ref != None and torch.any((body_model_output.right_hand_pose).detach().cpu() > 0):
            if torch.median(joints_conf[:, 46:67]) == 0.0 and not self.use_symmetry:
                reference_weight = self.right_reference_weight * 10
            else:
                reference_weight = self.right_reference_weight
            right_handpose_loss = L2Loss(body_model_output.right_hand_pose,
                                         self.right_hand_ref,
                                         (15, 3), reference_weight)

        left_handpose_loss = 0.0
        if self.left_hand_ref != None and torch.any((body_model_output.left_hand_pose).detach().cpu() > 0):
            if torch.median(joints_conf[:, 25:46]) == 0.0 and not self.use_symmetry:
                reference_weight = self.left_reference_weight * 10
            else:
                reference_weight = self.left_reference_weight
            left_handpose_loss = L2Loss(body_model_output.left_hand_pose,
                                        self.left_hand_ref,
                                        (15, 3), reference_weight)

        constraint_loss = symmetry_loss + right_handpose_loss + left_handpose_loss

        total_loss = temporal_loss + constraint_loss

        return total_loss
