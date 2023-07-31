import torch
import h5py
import numpy as np
import open3d as o3d
import os
import trimesh
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms as trans

from options.test_options import TestOptions
from data import DataLoader
from models import create_model
import utils.utils as utils

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def main(epoch=-1, name="", is_train=True):
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = name
    dataset = DataLoader(opt, is_train)
    model = create_model(opt)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    for i, data in enumerate(dataset):
        model.set_input(data)
        
        prediction, target, condfidence = model.test(vis=True)
        
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        condfidence = condfidence.cpu().detach().numpy()
        
        # prediction_cp = utils.transform_control_points_numpy(
        #     prediction, prediction.shape[0]
        # )
        
        # draw predicted grasp and gt grasp using open3d
        
        # grasp_rt = data['grasp_rt'].reshape(-1,4,4)
        # gt_cps = data['target_cps']
        # print('gt control points from dataset-----------------------------')
        # print(gt_cps[0])
        # for i in range(grasp_rt.shape[0]):
        #     homog_mat = grasp_rt[i]
        #     # print('homog_mat: ', homog_mat)

        #     rot_mat = homog_mat[:3,:3]
        #     t = homog_mat[:3,3]
        #     homog_mat = homog_mat.reshape(1,4,4)


        #     gt_control_points = utils.transform_control_points_numpy(
        #         homog_mat, 64, mode='rt', is_bimanual=True)
        #     print('gt control point by rotation matrix------------------------')
        #     print(gt_control_points[0])
            
        #     r = R.from_matrix(rot_mat)
        #     q = r.as_quat()

        #     rot_mat_torch = torch.tensor(rot_mat, dtype=torch.float64)
        #     rot_mat_torch = torch.transpose(rot_mat_torch, 0, 1)
        #     q_torch = trans.matrix_to_quaternion(rot_mat_torch)
        #     # rot_mat_from_qtorch = trans.quaternion_to_matrix(q_torch)
        #     print('--------------------')
        #     print('q from scipy: ', q)
        #     qt = np.concatenate([q_torch.cpu().numpy(),t], axis=0).reshape(1,7)
        #     # qt = np.concatenate([q,t], axis=0).reshape(1,7)
            
        #     qt = torch.tensor(qt, dtype=torch.float64)
        #     rs, ts = utils.convert_qt_to_rt(qt, is_bimanual=True)
        #     rot_mat_rs = utils.tc_rotation_matrix(rs[0, 0], rs[0, 1], rs[0, 2], batched=True)

        #     # rot_mat_q = R.from_quat(q)
        #     # rot_mat_q = rot_mat_q.as_matrix()
        #     # print(rot_mat)
        #     # print(rot_mat_q)
        #     # exit()

        #     gt_control_points_qt = utils.transform_control_points(qt.reshape(1,7).repeat(64,1), 64, mode='qt', is_bimanual=True)
        #     print('gt control point by quaternion-----------------------------')
        #     print(gt_control_points_qt[0])
        #     exit()
            
        
        prediction_grasps = []
        gt_grasps = []
        for i in range(prediction.shape[0]):
            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(data['pc'][0])
            
            
            target_grasp = target[i]
            pcd_target_grasp = o3d.geometry.PointCloud()
            pcd_target_grasp.points = o3d.utility.Vector3dVector(target_grasp)
            pcd_target_grasp.paint_uniform_color([0, 0, 0])
            gt_grasps.append(pcd_target_grasp)
            
            pcd_predict_grasp = o3d.geometry.PointCloud()
            pcd_predict_grasp.points = o3d.utility.Vector3dVector(prediction[i])
            pcd_predict_grasp.paint_uniform_color([1, 0, 0])
            prediction_grasps.append(pcd_predict_grasp)
            
            # o3d.visualization.draw_geometries([pcd_object]+[pcd_target_grasp]+[pcd_predict_grasp])
            # exit()
        o3d.visualization.draw_geometries([pcd_object]+gt_grasps)
        
        
def create_robotiq_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 0],
            [4.10000000e-02, -7.27595772e-12, 0.067500],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 0],
            [-4.100000e-02, -7.27595772e-12, 0.067500],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, -0.067500/2], [0, 0, 0]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-0.085/2, 0, 0], [0.085/2, 0, 0]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    # z axis to x axis
    R = np.array([[1,0,0],[0,0,1],[0,-1,0]]).reshape(3,3)
    t =  np.array([0, 0, 0]).reshape(3,1)
    #
    T = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
    tmp.apply_transform(T)

    return tmp



if __name__ == '__main__':
    main(name='vae_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2_bimanual_old')