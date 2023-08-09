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
    # model = create_model(opt)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    for i, data in enumerate(dataset):
        pc_np = data['pc']
        
        #* vis gt grasp is valid
        furthest_distance = np.max(np.sqrt(np.sum(abs(pc_np)**2,axis=-1)))
        print(furthest_distance)
        
        for i in range(pc_np.shape[0]):
            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(pc_np[i])
            pcd_object.paint_uniform_color([0.0, 0.0, 1.0])
            
            gt_grasp_point1 = data['target_cps1']
            gt_grasp_point2 = data['target_cps2']
            
            pcd_grasp1 = o3d.geometry.PointCloud()
            pcd_grasp2 = o3d.geometry.PointCloud()
            pcd_grasp1.points = o3d.utility.Vector3dVector(gt_grasp_point1[i])
            pcd_grasp2.points = o3d.utility.Vector3dVector(gt_grasp_point2[i])
            pcd_grasp1.paint_uniform_color([1.0, 0.0, 0.0])
            pcd_grasp2.paint_uniform_color([1.0, 0.0, 0.0])
            
            o3d.visualization.draw_geometries([pcd_object, pcd_grasp1, pcd_grasp2])
        exit()
        
        pc_mean = np.mean(pc_np[0], 0)
        pc_np -= np.expand_dims(pc_mean, 0)
        
        pc = torch.from_numpy(data['pc']).to(device)
        grasps, condfidence, z = model.generate_grasps(pc)
        if opt.is_bimanual==False and opt.is_bimanual_v2==True:
            grasps1 = grasps[:, 0, :]
            grasps2 = grasps[:, 1, :]
            grasps_eulers1, grasp_translation1 = utils.convert_qt_to_rt(grasps1, is_bimanual=True)
            grasps_eulers2, grasp_translation2 = utils.convert_qt_to_rt(grasps2, is_bimanual=True)
            
            grasp1 = rot_ang_trans_to_grasps_wo_refine(grasps_eulers1, grasp_translation1)
            grasp2 = rot_ang_trans_to_grasps_wo_refine(grasps_eulers2, grasp_translation2)
            
            gt_grasps1 = data['grasp_rt1']
            gt_grasps2 = data['grasp_rt2']
            gt_grasps1 = gt_grasps1.reshape(-1,4,4)
            gt_grasps2 = gt_grasps2.reshape(-1,4,4)
            
            grasps = []
            grasps_denormalized = []
            grasps.append(grasp1)
            grasps.append(grasp2)
            grasps.append(gt_grasps1)
            grasps.append(gt_grasps2)
            for i in range(len(grasps)):
                grasps_tmp = denormalize_grasps(grasps[i], pc_mean)
                grasps_denormalized.append(grasps_tmp)
            grasps = grasps_denormalized
            gripper_predict_color = [(1.0, 0.0, 0.0) for i in range(opt.num_grasps_per_object)]
            gripper_gt_color = [(0.0, 1.0, 0.0) for i in range(opt.num_grasps_per_object)]
            gripper_color = gripper_predict_color + gripper_gt_color
            
            mlab.figure(bgcolor=(1,1,1))
            draw_scene(data['pc'][0],
                       grasps=grasps,
                       gripper_color=gripper_color,
                       is_bimanual_v2=opt.is_bimanual_v2)
            print('close the window to continue to next object')
            mlab.show()
        else:
            grasps_eulers, grasp_translation = utils.convert_qt_to_rt(grasps, is_bimanual=opt.is_bimanual)
            grasps = rot_ang_trans_to_grasps_wo_refine(grasps_eulers, grasp_translation)

            gt_grasps = data['grasp_rt']
            gt_grasps = gt_grasps.reshape(-1,4,4)
            for i in range(gt_grasps.shape[0]):
                grasps.append(gt_grasps[i])

            grasps = denormalize_grasps(grasps, pc_mean)
            gripper_predict_color = [(1.0, 0.0, 0.0) for i in range(opt.num_grasps_per_object)]
            gripper_gt_color = [(0.0, 1.0, 0.0) for i in range(opt.num_grasps_per_object)]
            gripper_color = gripper_predict_color + gripper_gt_color

            mlab.figure(bgcolor=(1,1,1))
            draw_scene(data['pc'][0],
                    grasps=grasps,
                    gripper_color=gripper_color,
                    is_bimanual=opt.is_bimanual)
            print('close the window to continue to next object ...')
            mlab.show()
    ################################*
        # model.set_input(data)
        # if opt.is_bimanual:
        #     prediction, target, condfidence, grasp_rt, prediction_q = model.test(vis=True)
        #     grasp_rt = grasp_rt.cpu().detach().numpy()
        #     grasp_rt = grasp_rt.reshape(-1,4,4)
        #     # prediction_q = prediction_q.cpu().detach().numpy()
        # else:
        #     prediction, target, condfidence, grasp_rt, prediction_q = model.test(vis=True)
        #     grasp_rt = grasp_rt.cpu().detach().numpy()
        #     grasp_rt = grasp_rt.reshape(-1,4,4)
        
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