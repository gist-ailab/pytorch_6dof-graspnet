import os
import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
from utils.sample import Object
import random
import time
import copy
from autolab_core import RigidTransform
import h5py
try:
    from Queue import Queue
except:
    from queue import Queue


class GraspEvaluatorData(BaseDataset):
    def __init__(self, opt, ratio_positive=0.3, ratio_hardnegative=0.4):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.collision_hard_neg_queue = {}
        #self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.ratio_positive = self.set_ratios(ratio_positive)
        self.ratio_hardnegative = self.set_ratios(ratio_hardnegative)

    def set_ratios(self, ratio):
        if int(self.opt.num_grasps_per_object * ratio) == 0:
            return 1 / self.opt.num_grasps_per_object
        return ratio

    def __getitem__(self, index):
        path = self.paths[index]
        if self.opt.balanced_data:
            data = self.get_uniform_evaluator_data(path)
        else:
            data = self.get_nonuniform_evaluator_data(path)

        gt_control_points = utils.transform_control_points_numpy(
            data[1], self.opt.num_grasps_per_object, mode='rt')

        meta = {}
        meta['pc'] = data[0][:, :, :3]
        meta['grasp_rt'] = gt_control_points[:, :, :3]
        meta['labels'] = data[2]
        meta['quality'] = data[3]
        meta['pc_pose'] = data[4]
        meta['cad_path'] = data[5]
        meta['cad_scale'] = data[6]
        
        return meta

    def __len__(self):
        return self.size

    def get_uniform_evaluator_data(self, path, verify_grasps=False):
        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            path)

        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self.opt.batch_size
        output_cad_scales = np.asarray([cad_scale] * self.opt.batch_size,
                                       np.float32)

        num_positive = int(self.opt.batch_size * self.opt.ratio_positive)
        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_hard_negative = int(self.opt.batch_size *
                                self.opt.ratio_hardnegative)
        num_flex_negative = self.opt.batch_size - num_positive - num_hard_negative
        negative_clusters = self.sample_grasp_indexes(num_flex_negative,
                                                      neg_grasps,
                                                      neg_qualities)
        hard_neg_candidates = []
        # Fill in Positive Examples.

        for clusters, grasps, qualities in zip(
            [positive_clusters, negative_clusters], [pos_grasps, neg_grasps],
            [pos_qualities, neg_qualities]):
            for cluster in clusters:
                selected_grasp = grasps[cluster[0]][cluster[1]]
                selected_quality = qualities[cluster[0]][cluster[1]]
                hard_neg_candidates += utils.perturb_grasp(
                    selected_grasp,
                    self.collision_hard_neg_num_perturbations,
                    self.collision_hard_neg_min_translation,
                    self.collision_hard_neg_max_translation,
                    self.collision_hard_neg_min_rotation,
                    self.collision_hard_neg_max_rotation,
                )

        if verify_grasps:
            collisions, heuristic_qualities = utils.evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self.collision_hard_neg_queue or len(
                self.collision_hard_neg_queue[path]) < num_hard_negative:
            if path not in self.collision_hard_neg_queue:
                self.collision_hard_neg_queue[path] = []
            #hard negatives are perturbations of correct grasps.
            collisions, heuristic_qualities = utils.evaluate_grasps(
                hard_neg_candidates, obj_mesh)

            hard_neg_mask = collisions | (heuristic_qualities < 0.001)
            hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
            np.random.shuffle(hard_neg_indexes)
            for index in hard_neg_indexes:
                self.collision_hard_neg_queue[path].append(
                    (hard_neg_candidates[index], -1.0))
            random.shuffle(self.collision_hard_neg_queue[path])

        # Adding positive grasps
        for positive_cluster in positive_clusters:
            #print(positive_cluster)
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)

        # Adding hard neg
        for i in range(num_hard_negative):
            grasp, quality = self.collision_hard_neg_queue[path][i]
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        self.collision_hard_neg_queue[path] = self.collision_hard_neg_queue[
            path][num_hard_negative:]

        # Adding flex neg
        if len(negative_clusters) != num_flex_negative:
            raise ValueError(
                'negative clusters should have the same length as num_flex_negative {} != {}'
                .format(len(negative_clusters), num_flex_negative))

        for negative_cluster in negative_clusters:
            selected_grasp = neg_grasps[negative_cluster[0]][
                negative_cluster[1]]
            selected_quality = neg_qualities[negative_cluster[0]][
                negative_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(0)

        #self.change_object(cad_path, cad_scale)
        for iter in range(self.opt.num_grasps_per_object):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                pc, camera_pose, _ = self.change_object_and_render(
                    cad_path,
                    cad_scale,
                    thread_id=torch.utils.data.get_worker_info().id
                    if torch.utils.data.get_worker_info() else 0)
                output_pcs.append(pc)
                output_pc_poses.append(utils.inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)

        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales

    def get_nonuniform_evaluator_data(self, path, verify_grasps=False):

        pos_grasps, pos_qualities, neg_grasps, neg_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(
            path)
        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self.opt.num_grasps_per_object
        output_cad_scales = np.asarray(
            [cad_scale] * self.opt.num_grasps_per_object, np.float32)

        num_positive = int(self.opt.num_grasps_per_object *
                           self.ratio_positive)

        positive_clusters = self.sample_grasp_indexes(num_positive, pos_grasps,
                                                      pos_qualities)
        num_negative = self.opt.num_grasps_per_object - num_positive
        negative_clusters = self.sample_grasp_indexes(num_negative, neg_grasps,
                                                      neg_qualities)
        hard_neg_candidates = []
        # Fill in Positive Examples.
        for positive_cluster in positive_clusters:
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            selected_quality = pos_qualities[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)
            hard_neg_candidates += utils.perturb_grasp(
                selected_grasp,
                self.collision_hard_neg_num_perturbations,
                self.collision_hard_neg_min_translation,
                self.collision_hard_neg_max_translation,
                self.collision_hard_neg_min_rotation,
                self.collision_hard_neg_max_rotation,
            )

        if verify_grasps:
            collisions, heuristic_qualities = utils.evaluate_grasps(
                output_grasps, obj_mesh)
            for computed_quality, expected_quality, g in zip(
                    heuristic_qualities, output_qualities, output_grasps):
                err = abs(computed_quality - expected_quality)
                if err > 1e-3:
                    raise ValueError(
                        'Heuristic does not match with the values from data generation {}!={}'
                        .format(computed_quality, expected_quality))

        # If queue does not have enough data, fill it up with hard negative examples from the positives.
        if path not in self.collision_hard_neg_queue or self.collision_hard_neg_queue[
                path].qsize() < num_negative:
            if path not in self.collision_hard_neg_queue:
                self.collision_hard_neg_queue[path] = Queue()
            #hard negatives are perturbations of correct grasps.
            random_selector = np.random.rand()
            if random_selector < self.ratio_hardnegative:
                #print('add hard neg')
                collisions, heuristic_qualities = utils.evaluate_grasps(
                    hard_neg_candidates, obj_mesh)
                hard_neg_mask = collisions | (heuristic_qualities < 0.001)
                hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
                np.random.shuffle(hard_neg_indexes)
                for index in hard_neg_indexes:
                    self.collision_hard_neg_queue[path].put(
                        (hard_neg_candidates[index], -1.0))
            if random_selector >= self.ratio_hardnegative or self.collision_hard_neg_queue[
                    path].qsize() < num_negative:
                for negative_cluster in negative_clusters:
                    selected_grasp = neg_grasps[negative_cluster[0]][
                        negative_cluster[1]]
                    selected_quality = neg_qualities[negative_cluster[0]][
                        negative_cluster[1]]
                    self.collision_hard_neg_queue[path].put(
                        (selected_grasp, selected_quality))

        # Use negative examples from queue.
        for _ in range(num_negative):
            #print('qsize = ', self._collision_hard_neg_queue[file_path].qsize())
            grasp, quality = self.collision_hard_neg_queue[path].get()
            output_grasps.append(grasp)
            output_qualities.append(quality)
            output_labels.append(0)

        for iter in range(self.opt.num_grasps_per_object):
            if iter > 0:
                output_pcs.append(np.copy(output_pcs[0]))
                output_pc_poses.append(np.copy(output_pc_poses[0]))
            else:
                pc, camera_pose, _ = self.change_object_and_render(
                    cad_path,
                    cad_scale,
                    thread_id=torch.utils.data.get_worker_info().id
                    if torch.utils.data.get_worker_info() else 0)
                #self.change_object(cad_path, cad_scale)
                #pc, camera_pose, _ = self.render_random_scene()

                output_pcs.append(pc)
                output_pc_poses.append(utils.inverse_transform(camera_pose))

            output_grasps[iter] = camera_pose.dot(output_grasps[iter])

        output_pcs = np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        output_qualities = np.asarray(output_qualities, dtype=np.float32)
        output_pc_poses = np.asarray(output_pc_poses, dtype=np.float32)
        return output_pcs, output_grasps, output_labels, output_qualities, output_pc_poses, output_cad_paths, output_cad_scales

class BimanualGraspEvaluatorDataset(BaseDataset):
    def __init__(self, opt, ratio_positive=0.3, ratio_hardnegative=0.4, is_train=True):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.is_train = is_train
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.collision_hard_neg_queue = {}
        opt.input_nc = self.ninput_channels
        self.ratio_positive = self.set_ratios(ratio_positive)
        self.ratio_hardnegative = self.set_ratios(ratio_hardnegative)
        
    
    def __len__(self):
        return self.size
    
    
    def __getitem__(self, index):
        path = self.paths[index]
        data = self.get_bimanual_evaluator_data(path)    
        
    
    def make_dataset(self):
        files = []
        file_list = os.listdir(os.path.join(self.opt.dataset_root_folder,
                               'grasps_processed'))
        files = [os.path.join(self.opt.dataset_root_folder, 'grasps_processed', file) for file in file_list]
        
        if not self.is_train:
            files = files[200:250]
        else:
            files = files[:200]

        return files
    
    
    def set_ratios(self, ratio):
        if int(self.opt.num_grasps_per_object * ratio) == 0:
            return 1 / self.opt.num_grasps_per_object
        return ratio
    
    
    def get_bimanual_evaluator_data(self, path, verify_grasps=False):
        pos_grasps, pos_qualities, obj_mesh, cad_path, cad_scale = self.read_grasp_file(path)
        output_pcs = []
        output_grasps = []
        output_qualities = []
        output_labels = []
        output_pc_poses = []
        output_cad_paths = [cad_path] * self.opt.batch_size
        output_cad_scales = np.asarray([cad_scale] * self.opt.batch_size,
                                       np.float32)
        num_positive = int(self.opt.num_grasps_per_object *
                           self.ratio_positive)
        num_negative = self.opt.num_grasps_per_object - num_positive
        sampled_grasp_idxs = np.random.choice(range(len(pos_grasps)), self.opt.num_grasps_per_object)
        
        # Fill in positive grasp data
        hard_neg_candidates = []
        for idx in sampled_grasp_idxs:
            selected_grasp = pos_grasps[idx]
            selected_quality = pos_qualities[idx]
            output_grasps.append(selected_grasp)
            output_qualities.append(selected_quality)
            output_labels.append(1)
            hard_neg_candidates += utils.perturb_grasp(
                selected_grasp,
                self.collision_hard_neg_num_perturbations,
                self.collision_hard_neg_min_translation,
                self.collision_hard_neg_max_translation,
                self.collision_hard_neg_min_rotation,
                self.collision_hard_neg_max_rotation,
            )
            
        if path not in self.collision_hard_neg_queue or self.collision_hard_neg_queue[path].qsize() < num_negative:
            if path not in self.collision_hard_neg_queue:
                self.collision_hard_neg_queue[path] = Queue()
            # add hard negative samples from positive grasps
            random_selector = np.random.rand()
            if random_selector < self.ratio_hardnegative:
                collisions, heuristic
            
        
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


        object_model.mesh.apply_transform(RigidTransform(np.eye(3), -object_model.mesh.centroid).matrix)
        object_model.rescale(mesh_scale)
        object_mean = object_model.mesh.centroid
        object_model = object_model.mesh        
        # load fileterd unique single grasp

        single_grasp = np.array(h5_file["grasps/single_grasps"])

        single_grasp[:, :3, 3] -= object_mean
        single_grasp_quality = np.array(h5_file["grasps/single_grasps_quality"])

        return single_grasp, single_grasp_quality, object_model, os.path.join(root_folder, mesh_root, mesh_fname), mesh_scale