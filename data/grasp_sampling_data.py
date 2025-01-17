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
        self.train_data_length = self.opt.train_data_length
        self.test_data_length = self.opt.test_data_length
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
            files = files[self.opt.train_data_length:self.opt.train_data_length+self.opt.test_data_length]
        else:
            files = files[:self.opt.train_data_length] #3315

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
        # tmp = copy.deepcopy(pos_qualities)
        # tmp = tmp.reshape(-1)
        # tmp = sorted(tmp, reverse=True)
        # pos_idx = len(tmp) // 10 * 4
        # pos_grasp_idx_list = np.where(np.isin(pos_qualities, tmp[:pos_idx]))[0]
        # if len(pos_grasp_idx_list) < self.opt.num_grasps_per_object:
        #     sampled_grasp_idxs = pos_grasp_idx_list
        #     while len(sampled_grasp_idxs) < self.opt.num_grasps_per_object:
        #         sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(pos_grasp_idx_list, 1))
        # else:
        #     sampled_grasp_idxs = np.random.choice(pos_grasp_idx_list, self.opt.num_grasps_per_object)
        

        # render the scene to get pc and camera pose using pyrender
        pc, camera_pose, _ = self.change_object_and_render(
            cad_path,
            cad_scale,
            thread_id=torch.utils.data.get_worker_info().id
            if torch.utils.data.get_worker_info() else 0)
        
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


class BimanualSecondGraspSamplingData(BaseDataset):
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
        pos_single_grasps, pos_bimanual_grasps, paired_idx_mapping, cad, cad_path, cad_scale = self.read_grasp_file(path)
        # pos_grasps, pos_qualities, _, cad_path, cad_scale = copy.deepcopy(self.cache[path])
        meta = {}
        #sample the grasp idx for data loader
        if len(pos_single_grasps) < self.opt.num_grasps_per_object:
            sampled_grasp_idxs = [i for i in range(len(pos_single_grasps))]
            while len(sampled_grasp_idxs) < self.opt.num_grasps_per_object:
                sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(len(pos_single_grasps), 1))
        else:
            sampled_grasp_idxs = np.random.choice(range(len(pos_single_grasps)), self.opt.num_grasps_per_object)
        # sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object)
        # tmp = copy.deepcopy(pos_qualities)
        # tmp = tmp.reshape(-1)
        # tmp = sorted(tmp, reverse=True)
        # pos_idx = len(tmp) // 10 * 4
        # pos_grasp_idx_list = np.where(np.isin(pos_qualities, tmp[:pos_idx]))[0]
        # if len(pos_grasp_idx_list) < self.opt.num_grasps_per_object:
        #     sampled_grasp_idxs = pos_grasp_idx_list
        #     while len(sampled_grasp_idxs) < self.opt.num_grasps_per_object:
        #         sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(pos_grasp_idx_list, 1))
        # else:
        #     sampled_grasp_idxs = np.random.choice(pos_grasp_idx_list, self.opt.num_grasps_per_object)
        

        # render the scene to get pc and camera pose using pyrender
        pc, camera_pose, _ = self.change_object_and_render(
            cad_path,
            cad_scale,
            thread_id=torch.utils.data.get_worker_info().id
            if torch.utils.data.get_worker_info() else 0)
        
        # get the grasp and quality for the sampled grasp idx
        output_qualities = []
        output_grasps = []
        output_paired_grasps = []
        for iter in range(self.opt.num_grasps_per_object):
            selected_grasp_index = sampled_grasp_idxs[iter]

            selected_grasp = pos_single_grasps[selected_grasp_index]
            # selected_quality = pos_qualities[selected_grasp_index]
            num_pair_grasps = len(paired_idx_mapping[selected_grasp_index])
            if num_pair_grasps < self.opt.num_grasps_per_object2:
                sampled_pair_grasp_idxs = [i for i in range(num_pair_grasps)]
                while len(sampled_pair_grasp_idxs) < self.opt.num_grasps_per_object2:
                    sampled_pair_grasp_idxs = np.append(sampled_pair_grasp_idxs, np.random.choice(num_pair_grasps, 1))
            else:
                sampled_pair_grasp_idxs = np.random.choice(range(num_pair_grasps), self.opt.num_grasps_per_object2)
            
            output_paired_grasps_tmp = []
            for iter2 in range(self.opt.num_grasps_per_object2):
                selected_grasp_index2 = sampled_pair_grasp_idxs[iter2]
                selected_pair_grasp = pos_bimanual_grasps[paired_idx_mapping[selected_grasp_index][selected_grasp_index2][0]][paired_idx_mapping[selected_grasp_index][selected_grasp_index2][1]]
                output_paired_grasps_tmp.append(camera_pose.dot(selected_pair_grasp))
            
            # output_qualities.append(selected_quality)
            
            # camera_pose = np.transpose(camera_pose)
            output_grasps.append(camera_pose.dot(selected_grasp)) #(batch_size, 4, 4)
            output_paired_grasps.append(output_paired_grasps_tmp) #(batch_size, batch_size2, 4, 4)
        # print(output_paired_grasps)
        # print(np.array(output_paired_grasps).shape)
        batch_size = self.opt.num_grasps_per_object*self.opt.num_grasps_per_object2
        output_grasps_tmp = np.zeros((batch_size, 4, 4))
        output_grasps_tmp[1::2] = np.array(output_grasps)
        output_grasps_tmp[::2] = np.array(output_grasps)
        output_grasps = output_grasps_tmp
        
        output_paired_grasps_tmp = np.zeros((batch_size, 4, 4))
        output_paired_grasps_tmp[::2] = np.array(output_paired_grasps)[:, 0, :, :]
        output_paired_grasps_tmp[1::2] = np.array(output_paired_grasps)[:, 1, :, :]
        output_paired_grasps = output_paired_grasps_tmp
        
        
        gt_control_points = utils.transform_control_points_numpy(
            output_grasps, batch_size, mode='rt', is_bimanual=self.opt.is_bimanual) #(batch_size, 6, 4)   
        gt_control_points_paired = utils.transform_control_points_numpy(
            output_paired_grasps, batch_size, mode='rt', is_bimanual=self.opt.is_bimanual) #(batch_size, 6, 4)

        
        meta['pc'] = np.array([pc] * batch_size)[:, :, :3]
        meta['grasp_rt'] = output_grasps.reshape(len(output_grasps), -1)
        meta['grasp_rt_paired'] = output_paired_grasps.reshape(len(output_paired_grasps), -1)
        meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] * batch_size)
        meta['cad_path'] = np.array([cad_path] * batch_size)
        meta['cad_scale'] = np.array([cad_scale] * batch_size)
        # meta['quality'] = np.array(output_qualities)
        meta['target_cps'] = np.array(gt_control_points[:, :, :3])
        meta['target_cps_paired'] = np.array(gt_control_points_paired[:, :, :3])
        return meta
    
    def __len__(self):
        return self.size
        

    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache:
            pos_grasps, pos_bimanual_grasps, paired_idx_mapping, cad, cad_path, cad_scale = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, pos_bimanual_grasps, paired_idx_mapping, cad, cad_path, cad_scale
        
        pos_grasps, pos_bimanual_grasps, paired_idx_mapping, cad, cad_path, cad_scale = self.read_object_grasp_data(
            path,
            ratio_of_grasps_to_be_used=self.opt.grasps_ratio,
            return_all_grasps=return_all_grasps)

        
        if self.caching:
            self.cache[file_name] = (pos_grasps, pos_bimanual_grasps, paired_idx_mapping, cad, cad_path, cad_scale)
            return copy.deepcopy(self.cache[file_name])
        
        return pos_grasps, pos_bimanual_grasps, paired_idx_mapping, cad, cad_path, cad_scale

    
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
        
        force_closure = np.array(h5_file["grasps/qualities/Force_closure"])
        torque_optimization = np.array(h5_file["grasps/qualities/Torque_optimization"])
        dexterity = np.array(h5_file["grasps/qualities/Dexterity"])
        
        force_closure_weight = 0.4
        dexterity_weight = 0.5
        torque_optimization_weight = 0.1
        
        sum_quality = force_closure_weight * force_closure + torque_optimization_weight * torque_optimization + \
                        dexterity_weight * dexterity
        sum_quality = sum_quality.reshape(-1)
        pos_quality_idx = np.where(sum_quality.reshape(-1) > 0.92)[0]
        
        grasps = np.asarray(h5_file["grasps/transforms"])
        bimanual_grasps = grasps[pos_quality_idx]
        bimanual_grasps[:, :, :3, 3] -= object_mean
        sum_quality = sum_quality[pos_quality_idx]
        
        single_grasp_quality = np.array(h5_file["grasps/single_grasps_quality"])
        paired_idx_mapping = eval(h5_file["grasps/paired_idx_mapping"][()])

        return single_grasp, bimanual_grasps, paired_idx_mapping, object_model, os.path.join(root_folder, mesh_root, mesh_fname), mesh_scale


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
            files_proccessed = files_proccessed[10:12]
        else:
            files_proccessed = files_proccessed[:10]

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
            pos_gt_control_points1 = []
            pos_gt_control_points2 = []
            pos_output_grasp1 = []
            pos_output_grasp2 = []
            pos_output_quality = []    

            pos_grasps = pos_grasps[pos_grasp_idxs]
            pos_qualities = pos_qualities[pos_grasp_idxs]
            output_grasps1 = camera_pose.dot(pos_grasps[:, 0, :, :])
            output_grasps1 = output_grasps1.transpose(1, 0, 2)
            output_grasps2 = camera_pose.dot(pos_grasps[:, 1, :, :])
            output_grasps2 = output_grasps2.transpose(1, 0, 2)
            
            gt_control_points1 = utils.transform_control_points_numpy(
                np.array(output_grasps1), len(output_grasps1), mode='rt', is_bimanual_v2=self.opt.is_bimanual_v2)
            gt_control_points2 = utils.transform_control_points_numpy(
                np.array(output_grasps2), len(output_grasps2), mode='rt', is_bimanual_v2=self.opt.is_bimanual_v2) #(2001, 6, 4)
            
            middle_points1 = (gt_control_points1[:, 4, :3]+gt_control_points1[:, 5, :3])/2
            middle_points2 = (gt_control_points2[:, 4, :3]+gt_control_points2[:, 5, :3])/2
            
            pc_tmp = np.expand_dims(pc, axis=0)
            middle_points1 = np.expand_dims(middle_points1, axis=1)
            middle_points2 = np.expand_dims(middle_points2, axis=1)
            dist1 = np.min(np.linalg.norm(pc_tmp[:, :, :3] - middle_points1, axis=2), axis=1)
            dist2 = np.min(np.linalg.norm(pc_tmp[:, :, :3] - middle_points2, axis=2), axis=1)

            banned_idxs = np.where((dist1 > 0.1) | (dist2 > 0.1))[0]
            pos_grasp_idxs = range(len(pos_grasps))
            pos_grasp_idxs = np.delete(pos_grasp_idxs, banned_idxs)

            final_grasps_idxs = pos_grasp_idxs
            if len(final_grasps_idxs) == 0:
                print('no grasp near partial point cloud')
                print('again rendering')
                print('cad_path', cad_path)
                print('-------------------')
                continue
            
            sampled_grasp_idxs = np.random.choice(pos_grasp_idxs, self.opt.num_grasps_per_object)
            output_grasps1 = output_grasps1[sampled_grasp_idxs]
            output_grasps2 = output_grasps2[sampled_grasp_idxs]
            gt_control_points1 = gt_control_points1[sampled_grasp_idxs]
            gt_control_points2 = gt_control_points2[sampled_grasp_idxs]
            output_qualities = pos_qualities[sampled_grasp_idxs]
        

            
        # while len(pos_gt_control_points1) < self.opt.num_grasps_per_object:
        #     output_qualities = []
        #     output_grasps1 = []
        #     output_grasps2 = []
        #     banned_grasp_idxs = []
            
        #     target_num_grasps = self.opt.num_grasps_per_object - len(pos_gt_control_points1)
        #     pos_grasp_idxs = np.delete(pos_grasp_idxs, banned_grasp_idxs)
        #     sampled_grasp_idxs = np.random.choice(pos_grasp_idxs, target_num_grasps)
            
        #     for iter in range(target_num_grasps):
        #         selected_grasp_index = sampled_grasp_idxs[iter]

        #         selected_grasp = pos_grasps[selected_grasp_index]
        #         selected_quality = pos_qualities[selected_grasp_index]
        #         output_qualities.append(selected_quality)

        #         output_grasps1.append(camera_pose.dot(selected_grasp[0])) #(64, 4, 4)
        #         output_grasps2.append(camera_pose.dot(selected_grasp[1]))
                
        #     gt_control_points1 = utils.transform_control_points_numpy(
        #         np.array(output_grasps1), target_num_grasps, mode='rt', is_bimanual_v2=self.opt.is_bimanual_v2) #(64, 6, 4)
        #     gt_control_points2 = utils.transform_control_points_numpy(
        #         np.array(output_grasps2), target_num_grasps, mode='rt', is_bimanual_v2=self.opt.is_bimanual_v2) #(64, 6, 4)
            
        #     # check if the grasp point is near the object point cloud
        #     # if not, sample another grasp point
        #     for i in range(target_num_grasps):
        #         target_cps1 = gt_control_points1[i][:, :3]
        #         target_cps2 = gt_control_points2[i][:, :3]
        #         middle_point1 = (target_cps1[4] + target_cps1[5]) / 2
        #         middle_point2 = (target_cps2[4] + target_cps2[5]) / 2
                
        #         # check if the middle point1 is near the object point cloud by threashold distacne
        #         dist1 = np.linalg.norm(pc[:, :3] - middle_point1, axis=1)
        #         dist2 = np.linalg.norm(pc[:, :3] - middle_point2, axis=1)
        #         if np.min(dist1) > 0.1 or np.min(dist2) > 0.1:
        #             banned_grasp_idxs.append(selected_grasp_index)
        #             # print('dist1 or dist2 is too far')
        #             continue
        #         else:
        #             pos_gt_control_points1.append(target_cps1)
        #             pos_gt_control_points2.append(target_cps2)
        #             pos_output_grasp1.append(output_grasps1[i])
        #             pos_output_grasp2.append(output_grasps2[i])
        #             pos_output_quality.append(output_qualities[i])
            
            
            
        # output_grasps1 = copy.deepcopy(np.asarray(pos_output_grasp1))
        # output_grasps2 = copy.deepcopy(np.asarray(pos_output_grasp2))
        # output_qualities = copy.deepcopy(np.asarray(pos_output_quality))
        # gt_control_points1 = copy.deepcopy(np.asarray(pos_gt_control_points1))
        # gt_control_points2 = copy.deepcopy(np.asarray(pos_gt_control_points2))
        
        
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
        self.train_data_length = self.opt.train_data_length
        self.test_data_length = self.opt.test_data_length
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
            sum_quality_idx = np.where(sum_quality.reshape(-1) > 0.92)[0]

            if len(sum_quality_idx) == 0:
                print('no grasp quality is over 0.92')
                continue
            
            files_proccessed.append(file)  
            
        if not self.is_train:
            files_proccessed = files_proccessed[self.train_data_length:self.train_data_length+self.test_data_length]
        else:
            files_proccessed = files_proccessed[:self.train_data_length]


        return files_proccessed 
    
    def __getitem__(self, index):
        path = self.paths[index]
        pos_grasps, pos_qualities, object_model, cad_path, cad_scale = self.read_grasp_file(path)
        meta = {}
        
        #sample grasp idx for data loading
        if pos_grasps.shape[0] < self.opt.num_grasps_per_object:
            sampled_grasp_idxs = range(len(pos_grasps))
            sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object - len(pos_grasps), replace=True))
        else:
            sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object, replace=False)

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
        pc = pc.astype(np.float32)
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
        sum_quality_positive_idx = np.where(sum_quality.reshape(-1) > 0.92)[0]
        pos_grasps = grasps[sum_quality_positive_idx]
        pos_quality = sum_quality[sum_quality_positive_idx]
        
        return pos_grasps, pos_quality, object_model, os.path.join(root_folder, mesh_root, mesh_fname), mesh_scale
    
class BimanualBlockGraspSamplingData(BaseDataset):
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
                               'grasps_processed'))
        files = [os.path.join(self.opt.dataset_root_folder, 'grasps_processed', file) for file in file_list]
        
        if not self.is_train:
            files = files[100:120]
        else:
            files = files[:100] #3315

        return files
    
    def __getitem__(self, index):
        path = self.paths[index]
        pos_grasps, pos_qualities, object_model, cad_path, cad_scale = self.read_grasp_file(path)
        meta = {}
        
        #sample grasp idx for data loading
        sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object, replace=False)

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
        pc = pc.astype(np.float32)
        
        cell_bounds = np.linspace(object_model.bounds[0,:]-0.1, object_model.bounds[1,:]+0.1, 4)
        diag_center = np.zeros((len(cell_bounds)-1, 3))
        cell_centers = []
        for i in range(len(cell_bounds)-1):
            diag_center[i] = (cell_bounds[i+1,:] + cell_bounds[i, :])/2
        for x in diag_center[:, 0]:
            for y in diag_center[:, 1]:
                for z in diag_center[:, 2]:
                    cell_centers.append([x, y, z])
        submesh_list = []
        bbox_list = []
        pointcloud_idxs = {}
        block_sampled_grasp_points = {}
        block_sampled_grasps = {}
        output_grasps = {}
        output_grasps_points = {}
        block_features = {}
        
        for i, cell_center in enumerate(cell_centers):
            bbox = trimesh.creation.box(extents=(cell_bounds[1,:]-cell_bounds[0,:] ), transform=RigidTransform(np.eye(3), cell_center).matrix)
            #check which point in bbox and get index
            idxs = bbox.contains(pc)
            inlier_pc_idxs = np.where(idxs==True)[0]
            # print(inlier_pc_idxs)
            if len(inlier_pc_idxs) == 0:
                continue
            # print(idxs)
            pointcloud_idxs[i] = inlier_pc_idxs

            # min_xyz = bbox.bounds[0,:]
            # max_xyz = bbox.bounds[1,:]
            # bbox_minmax[i] = np.array([min_xyz, max_xyz])
            
            # submesh = object_model.slice_plane(bbox.facets_origin, -bbox.facets_normal)
            # if len(submesh.vertices) == 0:
            #     continue
            # bbox_list.append(bbox)
            # submesh_list.append(submesh)
        
        # trimesh.Scene(bbox_list).show(flags={'wireframe': True})
        # trimesh.Scene(bbox_list+[object_model]).show(flags={'wireframe': True})
        # for i in range(len(submesh_list)):
        #     trimesh.Scene([submesh_list[i]]).show()
        
        # print(bbox_minmax)
        # print(bbox_minmax[0])
        # print(bbox_minmax[0][0])
        
        #* sample grasp for each bbox cell
        gt_control_points = utils.transform_control_points_numpy(
            np.array(pos_grasps), len(pos_grasps), mode='rt', is_bimanual=True)
        gt_grasp_points = 0.5*(gt_control_points[:, 2, :3] + gt_control_points[:, 3, :3]) #(num_gt_grasps, 3)
        # check if the grasp point is near the each cell's object point cloud
        # if grasp point is near the each cell's object point cloud, allocate grasp point to each cell
        pointcloud_idxs_keys = pointcloud_idxs.keys()
        remove_keys = []
        for key in pointcloud_idxs_keys:
            block_sampled_pc = pc[pointcloud_idxs[key]]
            dist_grasp_pc = np.expand_dims(gt_grasp_points, axis=1) - np.expand_dims(block_sampled_pc, axis=0)
            dist_grasp_pc = np.linalg.norm(dist_grasp_pc, axis=2)

            min_dist_grasp_pc = np.min(dist_grasp_pc, axis=1)
            block_sampled_grasp_idx = np.where(min_dist_grasp_pc <= 0.0675)[0]
            # print(block_sampled_grasp_idx)
            if len(block_sampled_grasp_idx) == 0:
                # pointcloud_idxs.pop(key)
                remove_keys.append(key)
                # print('no grasp near the block sampled point cloud')
                continue
            
            block_sampled_grasp_points[key] = gt_control_points[block_sampled_grasp_idx][:, :, :3]
            block_sampled_grasps[key] = pos_grasps[block_sampled_grasp_idx]
        
        #* remove keys from pointcloud_idxs in remove_keys list
        list(map(pointcloud_idxs.pop, remove_keys))

        #* visualize
        # for key in pointcloud_idxs.keys():
        #     pcd = o3d.geometry.PointCloud()
        #     pcd.points = o3d.utility.Vector3dVector(pc)
        #     pcd.paint_uniform_color([0, 0, 0])
            
        #     block_sampled_pcd = o3d.geometry.PointCloud()
        #     block_sampled_pcd.points = o3d.utility.Vector3dVector(pc[pointcloud_idxs[key]])
        #     block_sampled_pcd.paint_uniform_color([0, 1, 0])
            
        #     block_sampled_grasp_point_pcd = o3d.geometry.PointCloud()
        #     block_sampled_grasp_point_pcd.points = o3d.utility.Vector3dVector(block_sampled_grasp_points[key].reshape(-1, 3))
        #     block_sampled_grasp_point_pcd.paint_uniform_color([1, 0, 0])
        #     o3d.visualization.draw_geometries([pcd, block_sampled_pcd, block_sampled_grasp_point_pcd])
        
        #* sample grsap and grasp control point for each cell
        output_grasps = []
        output_grasps_points = []
        block_features = []
        for key in pointcloud_idxs.keys():
            if len(block_sampled_grasps[key]) > self.opt.num_grasps_per_object:
                sampled_grasp_idxs = np.random.choice(range(len(block_sampled_grasps[key])), self.opt.num_grasps_per_object, replace=False)
            else:
                sampled_grasp_idxs = range(len(block_sampled_grasps[key]))
                while len(sampled_grasp_idxs) < self.opt.num_grasps_per_object:
                    sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(range(len(block_sampled_grasps[key])), self.opt.num_grasps_per_object-len(sampled_grasp_idxs), replace=True))
            features = self.create_block_sampled_features(self.opt.num_grasps_per_object, self.opt.npoints, pointcloud_idxs[key])

            # output_grasps[key] = block_sampled_grasps[key][sampled_grasp_idxs]
            # output_grasps_points[key] = block_sampled_grasp_points[key][sampled_grasp_idxs]
            
            output_grasps.append(block_sampled_grasps[key][sampled_grasp_idxs])
            output_grasps_points.append(block_sampled_grasp_points[key][sampled_grasp_idxs])
            block_features.append(features)
        output_grasps = np.array(output_grasps)
        output_grasps_points = np.array(output_grasps_points).transpose(1, 0, 2, 3)
        block_features = np.array(block_features)

        
        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)[:, :, :3]

        meta['features'] = block_features

        meta['grasp_rt'] = output_grasps.reshape(output_grasps.shape[0], output_grasps.shape[1], -1)
        # meta['grasp_rt2'] = np.array(output_grasps2).reshape(
        #     len(output_grasps2), -1)
        # meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
        #                            self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        # meta['quality'] = np.array(output_qualities)
        meta['target_cps'] = output_grasps_points
        # meta['target_cps2'] = np.array(gt_control_points2[:, :, :3])
        return meta
        
        
    def __len__(self):
        return self.size
    
    def create_block_sampled_features(self, batch_size, pc_npoints, block_sampled_pointcloud_idx):
        features = np.zeros((batch_size, pc_npoints, 1))
        # for i in range(batch_size):
        features[:, block_sampled_pointcloud_idx, 0] = 1
        return features
    
    
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
    
class BimanualAnchorGraspSamplingData(BaseDataset):
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
        sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object, replace=False)

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
        pc = pc.astype(np.float32)
        
        output_grasps1 = pos_grasps[:, 0, :, :]
        # output_grasps2 = pos_grasps[:, 1, :, :]
        # output_grasps1 = output_grasps1[:, 3, :3]
        # output_grasps1[:, :3, 3] = output_grasps1[:, :3, 3] - object_mesh_mean
        # output_grasps2[:, :3, 3] = output_grasps2[:, :3, 3] - object_mesh_mean
        
        gt_control_points1 = utils.transform_control_points_numpy(
            np.array(output_grasps1), len(output_grasps1), mode='rt', is_bimanual_v2=True)
        # gt_control_points2 = utils.transform_control_points_numpy(
        #     np.array(output_grasps2), len(output_grasps2), mode='rt', is_bimanual_v2=True)
        
        #* get anchor point using only the first grasp

        anchor = gt_control_points1[:,0,:3] # (2001, 3)

        # compute point cloud close to anchor point less than threshold
        # dist = np.expand_dims(anchor, axis=2) - np.expand_dims(pc, axis=2)
        # print(np.expand_dims(np.expand_dims(anchor, axis=1), axis=2).shape)
        # print(np.expand_dims(np.expand_dims(pc, axis=0), axis=0).shape)
        
        # dist = np.expand_dims(np.expand_dims(anchor, axis=1), axis=2) -\
        #     np.expand_dims(np.expand_dims(pc, axis=0), axis=0)
        # dist = np.squeeze(dist)
        # dist = np.linalg.norm(dist, axis=2)
        # partial_pc = []
        # for i in range(len(dist)):
        #     close_pc_idx = np.where(dist[i]<0.3)[0]
        #     if len(close_pc_idx) > 256:
        #         pc_sample_idx = np.random.choice(range(len(close_pc_idx)), 256, replace=False)
        #         close_pc_idx = pc_sample_idx
        #     else:
        #         # print('not enough close points')
        #         pc_sample_idx = range(len(close_pc_idx))
        #         while len(pc_sample_idx) > 256:
        #             pc_sample_idx = np.append(pc_sample_idx, np.random.choice(range(len(close_pc_idx)), 256-len(pc_sample_idx), replace=False))    
        #     partial_pc.append(pc[close_pc_idx])
        # partial_pc = np.asarray(partial_pc) #(2001, 256, 3)
        
        # for i in range(len(dist)):
        #     close_pc_idx = np.where(dist[i]<0.3)
        #     partial_pc = pc[close_pc_idx]
        #     pcd_pc = o3d.geometry.PointCloud()
        #     pcd_pc.points = o3d.utility.Vector3dVector(pc)
        #     pcd_pc.paint_uniform_color([0, 0, 0])
        #     pcd_partial_pc = o3d.geometry.PointCloud()
        #     pcd_partial_pc.points = o3d.utility.Vector3dVector(partial_pc)
        #     pcd_partial_pc.paint_uniform_color([0, 1, 0])
        #     pcd_anchor = o3d.geometry.PointCloud()
        #     pcd_anchor.points = o3d.utility.Vector3dVector(anchor[i].reshape(1,3))
        #     pcd_anchor.paint_uniform_color([1, 0, 0])
        #     o3d.visualization.draw_geometries([pcd_pc, pcd_partial_pc, pcd_anchor])
        
        output_grasps1 = output_grasps1[sampled_grasp_idxs]
        # output_grasps2 = output_grasps2[sampled_grasp_idxs]
        gt_control_points1 = gt_control_points1[sampled_grasp_idxs]
        # gt_control_points2 = gt_control_points2[sampled_grasp_idxs]
        output_qualities = pos_qualities[sampled_grasp_idxs]
        # partial_pc = partial_pc[sampled_grasp_idxs]
        anchor1 = anchor[sampled_grasp_idxs]
        
        meta['pc'] = np.array([pc] * self.opt.num_grasps_per_object)[:, :, :3]
        # meta['partial_pc'] = partial_pc
        meta['grasp_rt1'] = np.array(output_grasps1).reshape(
            len(output_grasps1), -1)
        # meta['grasp_rt2'] = np.array(output_grasps2).reshape(
        #     len(output_grasps2), -1)
        # meta['pc_pose'] = np.array([utils.inverse_transform(camera_pose)] *
        #                            self.opt.num_grasps_per_object)
        meta['cad_path'] = np.array([cad_path] *
                                    self.opt.num_grasps_per_object)
        meta['cad_scale'] = np.array([cad_scale] *
                                     self.opt.num_grasps_per_object)
        meta['quality'] = np.array(output_qualities)
        meta['target_cps1'] = np.array(gt_control_points1[:, :, :3])
        meta['anchor1'] = anchor1
        # meta['target_cps2'] = np.array(gt_control_points2[:, :, :3])
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
    

class BimanualCrossConditionGraspSamplingData(BaseDataset):
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
        self.train_data_length = self.opt.train_data_length
        self.test_data_length = self.opt.test_data_length
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
            sum_quality_idx = np.where(sum_quality.reshape(-1) > 0.92)[0]

            if len(sum_quality_idx) == 0:
                print('no grasp quality is over 0.92')
                continue
            
            files_proccessed.append(file)  
            
        if not self.is_train:
            files_proccessed = files_proccessed[self.train_data_length:self.train_data_length+self.test_data_length]
        else:
            files_proccessed = files_proccessed[:self.train_data_length]


        return files_proccessed 
    
    def __getitem__(self, index):
        path = self.paths[index]
        pos_grasps, pos_qualities, object_model, cad_path, cad_scale = self.read_grasp_file(path)
        meta = {}
        
        #sample grasp idx for data loading
        if pos_grasps.shape[0] < self.opt.num_grasps_per_object:
            sampled_grasp_idxs = range(len(pos_grasps))
            sampled_grasp_idxs = np.append(sampled_grasp_idxs, np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object - len(pos_grasps), replace=True))
        else:
            sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object, replace=False)

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
        pc = pc.astype(np.float32)
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
        sum_quality_positive_idx = np.where(sum_quality.reshape(-1) > 0.92)[0]
        pos_grasps = grasps[sum_quality_positive_idx]
        pos_quality = sum_quality[sum_quality_positive_idx]
        
        return pos_grasps, pos_quality, object_model, os.path.join(root_folder, mesh_root, mesh_fname), mesh_scale