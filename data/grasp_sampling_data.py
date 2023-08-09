import os
import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
from utils.sample import Object
import copy
import h5py
import random
from time import time
from tqdm import tqdm
import open3d as o3d
from autolab_core import RigidTransform
import trimesh

class GraspSamplingData(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        # print(self.size)
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.i = 0

    def __getitem__(self, index):
        path = self.paths[index]
        # print('index',index)
        # print('grasp path is >>>>>>>',path)
        pos_grasps, pos_qualities, _, _, _, cad_path, cad_scale = self.read_grasp_file(
            path)
        meta = {}
        # print(meta)
        try:
            all_clusters = self.sample_grasp_indexes(
                self.opt.num_grasps_per_object, pos_grasps, pos_qualities)

        except NoPositiveGraspsException:
            if self.opt.skip_error:
                return None
            else:
                return self.__getitem__(np.random.randint(0, self.size))

        #self.change_object(cad_path, cad_scale)
        #pc, camera_pose, _ = self.render_random_scene()
        pc, camera_pose, _ = self.change_object_and_render(
            cad_path,
            cad_scale,
            thread_id=torch.utils.data.get_worker_info().id
            if torch.utils.data.get_worker_info() else 0)
        #* check point cloud is normalized
        # print(np.mean(pc, axis=0))

        # furthest_distance = np.max(np.sqrt(np.sum(abs(pc[:, :3])**2,axis=-1)))
        # print(furthest_distance)
        # furthest_distance = np.max(np.sqrt(abs(pc)**2))
        # print(furthest_distance)
        # exit()
        
        output_qualities = []
        output_grasps = []
        for iter in range(self.opt.num_grasps_per_object):
            selected_grasp_index = all_clusters[iter]

            selected_grasp = pos_grasps[selected_grasp_index[0]][
                selected_grasp_index[1]]
            selected_quality = pos_qualities[selected_grasp_index[0]][
                selected_grasp_index[1]]
            output_qualities.append(selected_quality)
            output_grasps.append(camera_pose.dot(selected_grasp))

        gt_control_points = utils.transform_control_points_numpy(
            np.array(output_grasps), self.opt.num_grasps_per_object, mode='rt')

        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)[:, :, :3]
        meta['grasp_rt'] = np.array(output_grasps).reshape(
            len(output_grasps), -1)

        meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
                                   self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        meta['quality'] = np.array(output_qualities)
        meta['target_cps'] = np.array(gt_control_points[:, :, :3])
        return meta

    def __len__(self):
        return self.size

class BimanualGraspSamplingData(BaseDataset):
    def __init__(self, opt, is_train=True):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        # self.paths = self.make_dataset()
        # self.size = len(self.paths)
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.is_train = is_train
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.i = 0
        self.cache = {}
        # load all the grasp data before training
        # print('>>>>>>>>>>>>>>>>loading all the grasp data')
        # for path in tqdm(self.paths):
        #     pos_grasps, pos_qualities, _, cad_path, cad_scale = self.read_grasp_file(path)
        #     self.cache[path] = copy.deepcopy((pos_grasps, pos_qualities, cad_path, cad_scale))
        
        
    def make_dataset(self):
        files = []
        file_list = os.listdir(os.path.join(self.opt.dataset_root_folder,
                               'grasps_processed'))
        files = [os.path.join(self.opt.dataset_root_folder, 'grasps_processed', file) for file in file_list]
        
        if not self.is_train:
            files = files[100:120]
        else:
            files = files[:100] #3315

        return files
    
    def __getitem__(self, index):
        path = self.paths[index]
        pos_grasps, pos_qualities, _, cad_path, cad_scale = self.read_grasp_file(path)
        # pos_grasps, pos_qualities, _, cad_path, cad_scale = copy.deepcopy(self.cache[path])
        meta = {}
        #sample the grasp idx for data loader
        if len(pos_grasps) < self.opt.num_grasps_per_object:
            sampled_grasp_idxs = [i for i in range(len(pos_grasps))]
            while len(sampled_grasp_idxs) < self.opt.num_grasps_per_object:
                sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(len(pos_grasps), 1))
        else:
            sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object)
        # sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object)
        tmp = copy.deepcopy(pos_qualities)
        tmp = tmp.reshape(-1)
        tmp = sorted(tmp, reverse=True)
        pos_idx = len(tmp) // 10 * 4
        pos_grasp_idx_list = np.where(np.isin(pos_qualities, tmp[:pos_idx]))[0]
        if len(pos_grasp_idx_list) < self.opt.num_grasps_per_object:
            sampled_grasp_idxs = pos_grasp_idx_list
            while len(sampled_grasp_idxs) < self.opt.num_grasps_per_object:
                sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(pos_grasp_idx_list, 1))
        else:
            sampled_grasp_idxs = np.random.choice(pos_grasp_idx_list, self.opt.num_grasps_per_object)
        

        # render the scene to get pc and camera pose using pyrender
        pc, camera_pose, _ = self.change_object_and_render(
            cad_path,
            cad_scale,
            thread_id=torch.utils.data.get_worker_info().id
            if torch.utils.data.get_worker_info() else 0)
        # print(np.mean(pc, axis=0))
        furthest_distance = np.max(np.sqrt(np.sum(abs(pc[:, :3])**2,axis=-1)))
        print(furthest_distance)
        # exit()
        # get the grasp and quality for the sampled grasp idx
        output_qualities = []
        output_grasps = []
        for iter in range(self.opt.num_grasps_per_object):
            selected_grasp_index = sampled_grasp_idxs[iter]

            selected_grasp = pos_grasps[selected_grasp_index]
            selected_quality = pos_qualities[selected_grasp_index]
            output_qualities.append(selected_quality)
            
            # camera_pose = np.transpose(camera_pose)
            output_grasps.append(camera_pose.dot(selected_grasp)) #(64, 4, 4)
        
        gt_control_points = utils.transform_control_points_numpy(
            np.array(output_grasps), self.opt.num_grasps_per_object, mode='rt', is_bimanual=self.opt.is_bimanual) #(64, 6, 4)

        
            
        
        
        
        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)[:, :, :3]
        meta['grasp_rt'] = np.array(output_grasps).reshape(
            len(output_grasps), -1)
        meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
                                   self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        meta['quality'] = np.array(output_qualities)
        meta['target_cps'] = np.array(gt_control_points[:, :, :3])
        return meta
    
    def __len__(self):
        return self.size
        
        
    
    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache:
            pos_grasps, pos_qualities, cad, cad_path, cad_scale = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, pos_qualities, cad, cad_path, cad_scale
        
        pos_grasps, pos_qualities, cad, cad_path, cad_scale = self.read_object_grasp_data(
            path,
            ratio_of_grasps_to_be_used=self.opt.grasps_ratio,
            return_all_grasps=return_all_grasps)

        
        if self.caching:
            self.cache[file_name] = (pos_grasps, pos_qualities, cad, cad_path, cad_scale)
            return copy.deepcopy(self.cache[file_name])
        
        return pos_grasps, pos_qualities, cad, cad_path, cad_scale

    
    def read_object_grasp_data(self, 
                               h5_path, 
                               quality=['Dexterity', 'Force_closure', 'Torque_optimization'], 
                               ratio_of_grasps_to_be_used=1, 
                               return_all_grasps=False):
        
        num_clusters = self.opt.num_grasp_clusters
        root_folder = self.opt.dataset_root_folder
        mesh_root = 'meshes'
        
        if num_clusters <= 0:
            raise NoPositiveGraspsException
        
        # read h5 grasp file
        h5_file = h5py.File(h5_path, 'r')
        mesh_fname = h5_file['object/file'][()]

        mesh_scale = h5_file['object/scale'][()]
        # load and rescale, translate object mesh
        object_model = Object(os.path.join(root_folder, mesh_root, mesh_fname))
        # object_model.rescale(mesh_scale)
        # object_model = object_model.mesh
        # object_mean = np.mean(object_model.vertices, 0, keepdims=1)
        # object_model.vertices -= object_mean

        object_model.mesh.apply_transform(RigidTransform(np.eye(3), -object_model.mesh.centroid).matrix)
        object_model.rescale(mesh_scale)
        object_mean = object_model.mesh.centroid
        object_model = object_model.mesh
        # trimesh.Scene(object_model).show() 
        # load bimanual grasp
        # grasps = np.asarray(h5_file['grasps/transforms'])
        # grasps[:, :, :3, 3] -= object_mean
        
        # scale each grasp quality and sum them up
        # force_closure = np.array(h5_file["grasps/qualities/Force_closure"])
        # torque_optimization = np.array(h5_file["grasps/qualities/Torque_optimization"])
        # dexterity = np.array(h5_file["grasps/qualities/Dexterity"])
        
        # force_closure_weight = 0.4
        # torque_optimization_weight = 0.5
        # dexterity_weight = 0.1
        
        # sum_quality = force_closure_weight * force_closure + torque_optimization_weight * torque_optimization + \
        #                 dexterity_weight * dexterity
        
        # filter bimanual grasp to unique single grasp and corresponding quality
        # single_grasp, single_grasp_quality = self.filter_single_grasp(sum_quality, grasps)
        single_grasp = np.array(h5_file["grasps/single_grasps"])

        single_grasp[:, :3, 3] -= object_mean
        single_grasp_quality = np.array(h5_file["grasps/single_grasps_quality"])

        return single_grasp, single_grasp_quality, object_model, os.path.join(root_folder, mesh_root, mesh_fname), mesh_scale
        
        
    def filter_single_grasp(self, sum_quality, grasps):
        first_grasp_candidate = np.unique(grasps[:, 0, :, :], axis=0)
        second_grasp_candidate = np.unique(grasps[:, 1, :, :], axis=0)
        total_grasp_candidate = np.concatenate((first_grasp_candidate, second_grasp_candidate), axis=0)
        unique_single_grasp_candidate = np.unique(total_grasp_candidate, axis=0)
        
        paired_first_grasp_quality = []
        for i in range(len(unique_single_grasp_candidate)):
            first_grasp = total_grasp_candidate[i]
            paired_grasp_idxs = self.find_paired_grasp(first_grasp, grasps)
            total_quality = 0
            for idx in paired_grasp_idxs:
                total_quality += sum_quality[idx]
            total_quality /= len(paired_grasp_idxs)
            paired_first_grasp_quality.append(total_quality)
        
        paired_first_grasp_quality = np.array(paired_first_grasp_quality)
            
        return unique_single_grasp_candidate, paired_first_grasp_quality
            
    
    
    def find_paired_grasp(self, first_grasp, bimanual_grasps):
        index_list = []
        grasp_transform = bimanual_grasps

        for i in range(len(grasp_transform)):

            if np.array_equal(first_grasp, grasp_transform[i][0]) or np.array_equal(first_grasp, grasp_transform[i][1]):
                index_list.append(i)
        
        return index_list


class BimanualGraspSamplingDataV2(BaseDataset):
    def __init__(self, opt, is_train=True):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        # self.paths = self.make_dataset()
        # self.size = len(self.paths)
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.is_train = is_train
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.i = 0

    def make_dataset(self):
        files = []
        file_list = os.listdir(os.path.join(self.opt.dataset_root_folder,
                               'grasps'))
        files = [os.path.join(self.opt.dataset_root_folder, 'grasps', file) for file in file_list]
        files_proccessed = []
        for file in files:
            h5_file = h5py.File(file, 'r')
            force_closure = np.array(h5_file["/grasps/qualities/Force_closure"])
            torque_optimization = np.array(h5_file["grasps/qualities/Torque_optimization"])
            dexterity = np.array(h5_file["grasps/qualities/Dexterity"])
            
            force_closure_weight = 0.4
            torque_optimization_weight = 0.1
            dexterity_weight = 0.5
            
            sum_quality = force_closure_weight * force_closure + torque_optimization_weight * torque_optimization + \
                            dexterity_weight * dexterity

            sum_quality = sum_quality.reshape(-1)
            sum_quality_idx = np.where(sum_quality.reshape(-1) > 0.85)[0]

            if len(sum_quality_idx) == 0:
                print('no grasp quality is over 0.85')
                continue
            
            files_proccessed.append(file)  
            
        if not self.is_train:
            files_proccessed = files_proccessed[100:120]
        else:
            files_proccessed = files_proccessed[:100]

        return files_proccessed
    
    def __getitem__(self, index):
        path = self.paths[index]
        pos_grasps, pos_qualities, _, cad_path, cad_scale = self.read_grasp_file(path)
        meta = {}
        # sample the grasp idx for data loader
        
        if len(pos_grasps) < self.opt.num_grasps_per_object:
            sampled_grasp_idxs = [i for i in range(len(pos_grasps))]
            while len(sampled_grasp_idxs) > self.opt.num_grasps_per_object:
                sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(len(pos_grasps), 1))
        else:
            sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object)
        # print(len(pos_grasps))
        # print(sampled_grasp_idxs)
        # exit()
        # sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object)
        
        final_grasps_idxs =[]
        while len(final_grasps_idxs) == 0:
            pos_grasp_idxs = np.where(pos_qualities.reshape(-1) > 0.85)[0]
        # render the scene to get pc and camera pose using pyrender
        pc, camera_pose, _ = self.change_object_and_render(
            cad_path,
            cad_scale,
            thread_id=torch.utils.data.get_worker_info().id
            if torch.utils.data.get_worker_info() else 0)
        
        # get the grasp and quality for the sampled grasp idx
        output_qualities = []
        output_grasps1 = []
        output_grasps2 = []
        for iter in range(self.opt.num_grasps_per_object):
            selected_grasp_index = sampled_grasp_idxs[iter]

            selected_grasp = pos_grasps[selected_grasp_index]
            selected_quality = pos_qualities[selected_grasp_index]
            output_qualities.append(selected_quality)

            output_grasps1.append(camera_pose.dot(selected_grasp[0])) #(64, 4, 4)
            output_grasps2.append(camera_pose.dot(selected_grasp[1]))
            
        gt_control_points1 = utils.transform_control_points_numpy(
            np.array(output_grasps1), self.opt.num_grasps_per_object, mode='rt', is_bimanual_v2=self.opt.is_bimanual_v2) #(64, 6, 4)
        gt_control_points2 = utils.transform_control_points_numpy(
            np.array(output_grasps2), self.opt.num_grasps_per_object, mode='rt', is_bimanual_v2=self.opt.is_bimanual_v2) #(64, 6, 4)
        
        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)[:, :, :3]
        meta['grasp_rt1'] = np.array(output_grasps1).reshape(
            len(output_grasps1), -1)
        meta['grasp_rt2'] = np.array(output_grasps2).reshape(
            len(output_grasps2), -1)
        meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
                                   self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        meta['quality'] = np.array(output_qualities)
        meta['target_cps1'] = np.array(gt_control_points1[:, :, :3])
        meta['target_cps2'] = np.array(gt_control_points2[:, :, :3])
        return meta
    
    def __len__(self):
        return self.size
        
        
    
    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache:
            pos_grasps, pos_qualities, cad, cad_path, cad_scale = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, pos_qualities, cad, cad_path, cad_scale
        # start = time()
        pos_grasps, pos_qualities, cad, cad_path, cad_scale = self.read_object_grasp_data(
            path,
            ratio_of_grasps_to_be_used=self.opt.grasps_ratio,
            return_all_grasps=return_all_grasps)
        # end = time()
        # print('>>>>>>>>>>>>>>>>data load time: ', end - start)
        if self.caching:
            self.cache[file_name] = (pos_grasps, pos_qualities, cad, cad_path, cad_scale)
            return copy.deepcopy(self.cache[file_name])
        
        return pos_grasps, pos_qualities, cad, cad_path, cad_scale

    
    def read_object_grasp_data(self, 
                               h5_path, 
                               quality=['Dexterity', 'Force_closure', 'Torque_optimization'], 
                               ratio_of_grasps_to_be_used=1, 
                               return_all_grasps=False):
        
        num_clusters = 16
        root_folder = self.opt.dataset_root_folder
        mesh_root = 'meshes'
        
        if num_clusters <= 0:
            raise NoPositiveGraspsException
        
        # read h5 grasp file
        h5_file = h5py.File(h5_path, 'r')
        mesh_fname = h5_file['object/file'][()]

        mesh_scale = h5_file['object/scale'][()]
        # load and rescale, translate object mesh
        object_model = Object(os.path.join(root_folder, mesh_root, mesh_fname))
        # object_model.rescale(mesh_scale)
        # object_model = object_model.mesh
        # object_mean = np.mean(object_model.vertices, 0, keepdims=1)
        # object_model.vertices -= object_mean
        
        object_model.mesh.apply_transform(RigidTransform(np.eye(3), -object_model.mesh.centroid).matrix)
        object_model.rescale(mesh_scale)
        object_mean = object_model.mesh.centroid
        object_model = object_model.mesh 
        # load bimanual grasp
        
        grasps = np.asarray(h5_file['grasps/transforms'])
        grasps[:, :, :3, 3] -= object_mean
        
        # scale each grasp quality and sum them up
        force_closure = np.array(h5_file["/grasps/qualities/Force_closure"])
        torque_optimization = np.array(h5_file["grasps/qualities/Torque_optimization"])
        dexterity = np.array(h5_file["grasps/qualities/Dexterity"])
        
        force_closure_weight = 0.4
        torque_optimization_weight = 0.1
        dexterity_weight = 0.5
        
        sum_quality = force_closure_weight * force_closure + torque_optimization_weight * torque_optimization + \
                        dexterity_weight * dexterity
        
        # filter bimanual grasp to unique single grasp and corresponding quality
        # single_grasp, single_grasp_quality = self.filter_single_grasp(sum_quality, grasps)
        
        # def cluster_grasps(grasps, qualities):
        #     cluster_indexes = np.asarray(
        #         utils.farthest_points(grasps, num_clusters,
        #                               utils.distance_by_translation_grasp))
        #     return 
        
        # if not return_all_grasps:
        #     positive_grasps, positive_qualities = cluster_grasps(grasps, sum_quality)
        
        return grasps, sum_quality, object_model, os.path.join(root_folder, mesh_root, mesh_fname), mesh_scale
        
        
    def filter_single_grasp(self, sum_quality, grasps):
        first_grasp_candidate = np.unique(grasps[:, 0, :, :], axis=0)
        second_grasp_candidate = np.unique(grasps[:, 1, :, :], axis=0)
        total_grasp_candidate = np.concatenate((first_grasp_candidate, second_grasp_candidate), axis=0)
        unique_single_grasp_candidate = np.unique(total_grasp_candidate, axis=0)
        
        paired_first_grasp_quality = []
        for i in range(len(unique_single_grasp_candidate)):
            first_grasp = total_grasp_candidate[i]
            paired_grasp_idxs = self.find_paired_grasp(first_grasp, grasps)
            total_quality = 0
            for idx in paired_grasp_idxs:
                total_quality += sum_quality[idx]
            total_quality /= len(paired_grasp_idxs)
            paired_first_grasp_quality.append(total_quality)
        
        paired_first_grasp_quality = np.array(paired_first_grasp_quality)
            
        return unique_single_grasp_candidate, paired_first_grasp_quality
            
    
    
    def find_paired_grasp(self, first_grasp, bimanual_grasps):
        index_list = []
        grasp_transform = bimanual_grasps

        for i in range(len(grasp_transform)):

            if np.array_equal(first_grasp, grasp_transform[i][0]) or np.array_equal(first_grasp, grasp_transform[i][1]):
                index_list.append(i)
        
        return index_list

class BimanualGraspSamplingDataV3(BaseDataset):
    def __init__(self, opt, is_train=True):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        # self.paths = self.make_dataset()
        # self.size = len(self.paths)
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.is_train = is_train
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.i = 0
        
    def make_dataset(self):
        files = []
        file_list = os.listdir(os.path.join(self.opt.dataset_root_folder,
                               'grasps'))
        files = [os.path.join(self.opt.dataset_root_folder, 'grasps', file) for file in file_list]
        files_proccessed = []
        for file in files:
            h5_file = h5py.File(file, 'r')
            force_closure = np.array(h5_file["/grasps/qualities/Force_closure"])
            torque_optimization = np.array(h5_file["grasps/qualities/Torque_optimization"])
            dexterity = np.array(h5_file["grasps/qualities/Dexterity"])
            
            force_closure_weight = 0.4
            dexterity_weight = 0.5
            torque_optimization_weight = 0.1
            
            
            sum_quality = force_closure_weight * force_closure + torque_optimization_weight * torque_optimization + \
                            dexterity_weight * dexterity

            sum_quality = sum_quality.reshape(-1)
            sum_quality_idx = np.where(sum_quality.reshape(-1) > 0.85)[0]

            if len(sum_quality_idx) == 0:
                print('no grasp quality is over 0.85')
                continue
            
            files_proccessed.append(file)  
            
        if not self.is_train:
            files_proccessed = files_proccessed[100:120]
        else:
            files_proccessed = files_proccessed[:100]

        return files_proccessed
    
    def __getitem__(self, index):
        path = self.paths[index]
        pos_grasps, pos_qualities, object_model, cad_path, cad_scale = self.read_grasp_file(path)
        meta = {}
        
        #sample grasp idx for data loading
        sampled_grasp_idxs = np.random.choice(len(pos_grasps), self.opt.num_grasps_per_object, replace=False)
        
        #* sample whole point cloud from mesh model
        # load trimesh object
        # object_mesh = trimesh.load(cad_path)
        # if isinstance(object_mesh, list):
        #     object_mesh = trimesh.util.concatenate(object_mesh)
        # # scale and center the object
        # object_mesh.apply_scale(cad_scale)
        # object_mesh_mean = np.mean(object_mesh.vertices, axis=0)
        # object_mesh.vertices -= object_mesh_mean

        # sample points from the object
        pc = object_model.sample(self.opt.npoints)
        output_grasps1 = pos_grasps[:, 0, :, :]
        output_grasps2 = pos_grasps[:, 1, :, :]
        # output_grasps1 = output_grasps1[:, 3, :3]
        # output_grasps1[:, :3, 3] = output_grasps1[:, :3, 3] - object_mesh_mean
        # output_grasps2[:, :3, 3] = output_grasps2[:, :3, 3] - object_mesh_mean
        
        gt_control_points1 = utils.transform_control_points_numpy(
            np.array(output_grasps1), len(output_grasps1), mode='rt', is_bimanual_v2=True)
        gt_control_points2 = utils.transform_control_points_numpy(
            np.array(output_grasps2), len(output_grasps2), mode='rt', is_bimanual_v2=True)
        
        output_grasps1 = output_grasps1[sampled_grasp_idxs]
        output_grasps2 = output_grasps2[sampled_grasp_idxs]
        gt_control_points1 = gt_control_points1[sampled_grasp_idxs]
        gt_control_points2 = gt_control_points2[sampled_grasp_idxs]
        output_qualities = pos_qualities[sampled_grasp_idxs]
        
        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)[:, :, :3]
        meta['grasp_rt1'] = np.array(output_grasps1).reshape(
            len(output_grasps1), -1)
        meta['grasp_rt2'] = np.array(output_grasps2).reshape(
            len(output_grasps2), -1)
        # meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
        #                            self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        meta['quality'] = np.array(output_qualities)
        meta['target_cps1'] = np.array(gt_control_points1[:, :, :3])
        meta['target_cps2'] = np.array(gt_control_points2[:, :, :3])
        return meta
        
        
    def __len__(self):
        return self.size
    
    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache:
            pos_grasps, pos_qualities, cad, cad_path, cad_scale = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, pos_qualities, cad, cad_path, cad_scale

        pos_grasps, pos_qualities, cad, cad_path, cad_scale = self.read_object_grasp_data(
            path,
            ratio_of_grasps_to_be_used=self.opt.grasps_ratio,
            return_all_grasps=return_all_grasps)

        if self.caching:
            self.cache[file_name] = (pos_grasps, pos_qualities, cad, cad_path, cad_scale)
            return copy.deepcopy(self.cache[file_name])
        
        return pos_grasps, pos_qualities, cad, cad_path, cad_scale
    
    def read_object_grasp_data(self, 
                               h5_path, 
                               quality=['Dexterity', 'Force_closure', 'Torque_optimization'], 
                               ratio_of_grasps_to_be_used=1, 
                               return_all_grasps=False):
        
        num_clusters = 16
        root_folder = self.opt.dataset_root_folder
        mesh_root = 'meshes'
        
        if num_clusters <= 0:
            raise NoPositiveGraspsException
        
        # read h5 grasp file
        h5_file = h5py.File(h5_path, 'r')
        mesh_fname = h5_file['object/file'][()]

        mesh_scale = h5_file['object/scale'][()]
        # load and rescale, translate object mesh
        object_model = Object(os.path.join(root_folder, mesh_root, mesh_fname))
        # object_model.rescale(mesh_scale)
        # object_model = object_model.mesh
        # object_mean = np.mean(object_model.vertices, 0, keepdims=1)
        # object_model.vertices -= object_mean
        
        object_model.mesh.apply_transform(RigidTransform(np.eye(3), -object_model.mesh.centroid).matrix)
        object_model.rescale(mesh_scale)
        object_mean = object_model.mesh.centroid
        object_model = object_model.mesh 
        # load bimanual grasp
        grasps = np.asarray(h5_file['grasps/transforms'])
        grasps[:, :, :3, 3] -= object_mean
        
        # scale each grasp quality and sum them up
        force_closure = np.array(h5_file["/grasps/qualities/Force_closure"])
        torque_optimization = np.array(h5_file["grasps/qualities/Torque_optimization"])
        dexterity = np.array(h5_file["grasps/qualities/Dexterity"])
        
        force_closure_weight = 0.4
        torque_optimization_weight = 0.1
        dexterity_weight = 0.5
        
        sum_quality = force_closure_weight * force_closure + torque_optimization_weight * torque_optimization + \
                        dexterity_weight * dexterity
                        
        return grasps, sum_quality, object_model, os.path.join(root_folder, mesh_root, mesh_fname), mesh_scale
        