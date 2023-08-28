import torch
import h5py
import numpy as np
import open3d as o3d
import os
import trimesh
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R
# import pytorch3d.transforms as trans
import mayavi.mlab as mlab
import json
from autolab_core import RigidTransform

from options.test_options import TestOptions
from data import DataLoader
from models import create_model
import utils.utils as utils
from utils.visualization_utils import *
from utils.sample import Object

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def main(epoch=-1, name="", is_train=True):
    
    
    #* vis object mesh file
    # original_root = 'unified_grasp_data'

    # original_json_path = os.listdir(os.path.join(original_root, 'grasps'))
    # original_json = json.load(open(os.path.join(os.path.join(original_root, 'grasps'),original_json_path[0])))
    # original_model = Object(os.path.join(original_root, original_json['object']))
    
    # original_model.rescale(original_json['object_scale'])
    # original_model = original_model.mesh
    # object_mean = np.mean(original_model.vertices, 0, keepdims=1)
    # original_model.vertices -= object_mean
    
    # vertices = original_model.vertices
    # faces = original_model.faces
    
    # o3d_mesh = o3d.geometry.TriangleMesh()
    # o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # bimanual_root = 'da2_dataset'
    # mesh_root = 'meshes'
    # bimanual_h5_path = os.listdir(os.path.join(bimanual_root, 'grasps_processed'))
    # bimanual_h5 = h5py.File(os.path.join(os.path.join(bimanual_root, 'grasps_processed'), bimanual_h5_path[0]), 'r')
    # bimanual_fname = bimanual_h5['object/file'][()]
    # bimanual_scale = bimanual_h5['object/scale'][()]
    # bimanaul_model = Object(os.path.join(bimanual_root, mesh_root, bimanual_fname))
    
    # bimanaul_model.mesh.apply_transform(RigidTransform(np.eye(3), -bimanaul_model.mesh.centroid).matrix)
    # bimanaul_model.rescale(bimanual_scale)
    # object_mean = bimanaul_model.mesh.centroid
    # bimanaul_model = bimanaul_model.mesh
    
    # vertices = bimanaul_model.vertices
    # faces = bimanaul_model.faces
    
    # o3d_bimanual_mesh = o3d.geometry.TriangleMesh()
    # o3d_bimanual_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # o3d_bimanual_mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # o3d.visualization.draw_geometries([o3d_mesh, o3d_bimanual_mesh])
    
    ##############################* vis sampling results
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = name
    print(is_train)
    dataset = DataLoader(opt, is_train)
    model = create_model(opt)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    for i, data in enumerate(dataset):
        pc_np = data['pc']
        
        #* vis gt grasp is valid
        # furthest_distance = np.max(np.sqrt(np.sum(abs(pc_np)**2,axis=-1)))
        # print(furthest_distance)
        
        # for i in range(pc_np.shape[0]):
        #     pcd_object = o3d.geometry.PointCloud()
        #     pcd_object.points = o3d.utility.Vector3dVector(pc_np[i])
        #     pcd_object.paint_uniform_color([0.0, 0.0, 1.0])
            
            # gt_grasp_point1 = data['target_cps1']
            # gt_grasp_point2 = data['target_cps2']
            
            # pcd_grasp1 = o3d.geometry.PointCloud()
            # pcd_grasp2 = o3d.geometry.PointCloud()
            # pcd_grasp1.points = o3d.utility.Vector3dVector(gt_grasp_point1[i])
            # pcd_grasp2.points = o3d.utility.Vector3dVector(gt_grasp_point2[i])
            # pcd_grasp1.paint_uniform_color([1.0, 0.0, 0.0])
            # pcd_grasp2.paint_uniform_color([1.0, 0.0, 0.0])
            
            # o3d.visualization.draw_geometries([pcd_object, pcd_grasp1, pcd_grasp2])
        #     o3d.visualization.draw_geometries([pcd_object])
        # exit()
        pc_mean = np.mean(pc_np[0], 0)
        pc_np -= np.expand_dims(pc_mean, 0)
        
        pc = torch.from_numpy(data['pc']).to(device)
        if opt.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, condfidence, z = model.generate_grasps(pc)
        else:
            grasps, condfidence, z = model.generate_grasps(pc)
            
        if opt.is_bimanual==False and opt.is_bimanual_v2==True:
            if opt.is_bimanual_v3:
                grasps_R1 = torch.stack([dir1, torch.cross(app1, dir1), app1], dim=2)
                grasps_t1 = point1 - 0.00675*app1
                homog_vec =  torch.tensor([0,0,0,1], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(grasps_R1.shape[0], 1, 1)
                grasp1 = torch.cat((torch.cat((grasps_R1, grasps_t1.unsqueeze(2)), dim=2), homog_vec), dim=1)
                grasp1 = grasp1.detach().cpu().numpy()
                
                grasps2_R2 = torch.stack([dir2, torch.cross(app2, dir2), app2], dim=2)
                grasps2_t2 = point2 - 0.00675*app2
                grasp2 = torch.cat((torch.cat((grasps2_R2, grasps2_t2.unsqueeze(2)), dim=2), homog_vec), dim=1)
                grasp2 = grasp2.detach().cpu().numpy()

                
            else:
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
        
        # prediction = prediction.cpu().detach().numpy()
        # target = target.cpu().detach().numpy()
        # condfidence = condfidence.cpu().detach().numpy()
        
        # prediction_cp = utils.transform_control_points_numpy(
        #     prediction, prediction.shape[0]
        # )
        
        # draw predicted grasp and gt grasp using open3d
        
        # grasp_rt = data['grasp_rt'].reshape(-1,4,4)
        # gt_cps = data['target_cps']
        # # print('gt control points from dataset-----------------------------')
        # # print(gt_cps[0])
        # for i in range(grasp_rt.shape[0]):
        #     homog_mat = grasp_rt[i]
        #     # print('homog_mat: ', homog_mat)

        #     rot_mat = homog_mat[:3,:3]
        #     t = homog_mat[:3,3]
        #     homog_mat = homog_mat.reshape(1,4,4)


        #     gt_control_points = utils.transform_control_points_numpy(
        #         homog_mat, 64, mode='rt', is_bimanual=opt.is_bimanual)
        #     print('gt control point by rotation matrix------------------------')
        #     print(gt_control_points[0])
        #     print('gt control point in dataset------------------------')
        #     print(gt_cps[i])
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
            
        #     qt = torch.tensor(qt, dtype=torch.float32)
        #     rs, ts = utils.convert_qt_to_rt(qt, is_bimanual=opt.is_bimanual)
        #     rot_mat_rs = utils.tc_rotation_matrix(rs[0, 0], rs[0, 1], rs[0, 2], batched=True)
        #     # print(rot_mat_rs)
        #     # print(rot_mat)
        #     # rot_mat_q = R.from_quat(q)
        #     # rot_mat_q = rot_mat_q.as_matrix()
        #     # print(rot_mat)
        #     # print(rot_mat_q)
        #     # exit()

        #     gt_control_points_qt = utils.transform_control_points(qt.reshape(1,7).repeat(64,1), 64, mode='qt', is_bimanual=opt.is_bimanual)
        #     print('gt control point by quaternion-----------------------------')
        #     print(gt_control_points_qt[0])
        #     exit()
            
        
        # prediction_grasps = []
        # gt_grasps = []
        # gt_mesh_grasps = []
        # prediction_mesh_grasps = []
        # for i in range(prediction.shape[0]):
        #     pcd_object = o3d.geometry.PointCloud()
        #     pcd_object.points = o3d.utility.Vector3dVector(data['pc'][i])
            
        #     prediction_grasp = prediction[i]
        #     grasps, condfidence = model.generate_grasps(data['pc'][i])
        #     grasps_eulers, grasp_translation = utils.convert_qt_to_rt(grasps, is_bimanual=opt.is_bimanual)
        #     grasps = rot_ang_trans_to_grasps_wo_refine(grasps_eulers, grasp_translation)
            
            
        #     target_grasp = target[i]
        #     pcd_target_grasp = o3d.geometry.PointCloud()
        #     pcd_target_grasp.points = o3d.utility.Vector3dVector(target_grasp)
        #     pcd_target_grasp.paint_uniform_color([0, 0, 0])
        #     gt_grasps.append(pcd_target_grasp)
            
        #     homog_mat = grasp_rt[i]
        #     robotiq_marker = create_robotiq_marker().apply_transform(homog_mat)
        #     # robotiq_marker_mesh = robotiq_marker.as_open3d
        #     robotiq_marker_vertices = robotiq_marker.vertices
        #     robotiq_marker_faces = robotiq_marker.faces
            
        #     robotiq_marker_mesh = o3d.geometry.TriangleMesh()
        #     robotiq_marker_mesh.vertices = o3d.utility.Vector3dVector(robotiq_marker_vertices)
        #     robotiq_marker_mesh.triangles = o3d.utility.Vector3iVector(robotiq_marker_faces)
        #     colors = [[0,0,1] for i in range(len(robotiq_marker_vertices))]
        #     robotiq_marker_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        #     gt_mesh_grasps.append(robotiq_marker_mesh)
        #     # robotiq_markder_points = robotiq_marker.vertices
        #     # robotiq_markder_points = trimesh.points.PointCloud(robotiq_markder_points)
        #     # o3d.visualization.draw_geometries([pcd_robotiq_marker])
            
            
            
        #     pcd_predict_grasp = o3d.geometry.PointCloud()
        #     pcd_predict_grasp.points = o3d.utility.Vector3dVector(prediction[i])
        #     pcd_predict_grasp.paint_uniform_color([1, 0, 0])
        #     prediction_grasps.append(pcd_predict_grasp)
            
        #     # print(prediction_q[i].shape)
        #     rs, ts = utils.convert_qt_to_rt(prediction_q[i].unsqueeze(0), is_bimanual=opt.is_bimanual, is_batched=True)
        #     rot_mat_rs = utils.tc_rotation_matrix(rs[:, 0], rs[:, 1], rs[:, 2], batched=True).permute(0,2,1)
        #     rot_mat = torch.cat((rot_mat_rs, ts.unsqueeze(-1)), dim=-1)
        #     pad_homog = torch.tensor([0, 0, 0, 1], dtype=torch.float64, device=device).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1)
        #     rot_mat = torch.cat((rot_mat, pad_homog), dim=1).cpu().detach().numpy() # (batch_size, 4, 4)      
            
        #     robotiq_marker_predict = create_robotiq_marker(color=[1,0,0]).apply_transform(rot_mat[0])
        #     robotiq_marker_predict_vertices = robotiq_marker_predict.vertices
        #     robotiq_marker_predict_faces = robotiq_marker_predict.faces
            
        #     robotiq_marker_predict_mesh = o3d.geometry.TriangleMesh()
        #     robotiq_marker_predict_mesh.vertices = o3d.utility.Vector3dVector(robotiq_marker_predict_vertices)
        #     robotiq_marker_predict_mesh.triangles = o3d.utility.Vector3iVector(robotiq_marker_predict_faces)
        #     colors = [[1,0,0] for i in range(len(robotiq_marker_predict_vertices))]
        #     robotiq_marker_predict_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        #     # o3d.visualization.draw_geometries([pcd_object]+[robotiq_marker_mesh]+[robotiq_marker_predict_mesh]
        #     #                                   +[pcd_target_grasp]+[pcd_predict_grasp])
        #     prediction_mesh_grasps.append(robotiq_marker_predict_mesh)
        #     # o3d.visualization.draw_geometries([pcd_object]+[pcd_target_grasp]+[pcd_predict_grasp])
        #     # exit()
        # # o3d.visualization.draw_geometries([pcd_object]+gt_grasps+prediction_grasps)
        # o3d.visualization.draw_geometries([pcd_object]+gt_mesh_grasps+prediction_mesh_grasps)
        # exit()
        
        
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
        height=1.0,
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 0],
            [-4.100000e-02, -7.27595772e-12, 0.067500],
        ],
        height=1.0,
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, -0.067500/2], [0, 0, 0]],height=1.0,
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-0.085/2, 0, 0], [0.085/2, 0, 0]],
        height=1.0,
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


def rot_ang_trans_to_grasps_wo_refine(euler_angles, translations):
    grasps = []
    translations = translations.detach().cpu().numpy()
    for i in range(euler_angles.shape[0]):
        rt = tra.euler_matrix(*euler_angles[i])
        rt[:3, 3] = translations[i]
        grasps.append(rt)
    return grasps


def denormalize_grasps(grasps, mean=0, std=1):
    for grasp in grasps:
        grasp[:3, 3] = (std * grasp[:3, 3] + mean)
    return grasps

if __name__ == '__main__':
    # main(name='/SSD3/Workspace/pytorch_6dof-graspnet/checkpoints/bengio/vae_lr_0002_bs_192_scale_1_npoints_128_radius_02_latent_size_5_bimanual_v2_bimanual_v3_kl_loss_weight_0.001_use_point_loss')
    main(name='/SSD3/Workspace/pytorch_6dof-graspnet/checkpoints/yeon/vae_lr_001_bs_512_scale_1_latent_size_5_bimanual_v2_bimanual_v3_kl_loss_weight_0.001_use_test_reparam')
    # main(name='vae_pretrained')