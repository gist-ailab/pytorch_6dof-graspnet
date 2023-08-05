import numpy as np
import os
import shutil
import h5py
from tqdm import tqdm
import shutil

data_root = 'unified_grasp_data/meshes'

#* for mug object in shapenetcore dataset
# data_path = 'shapenetcoremug'
# folder_list = os.listdir(os.path.join(data_root, data_path))
# # print(folder_list)
# for folder in folder_list:
#     if folder == 'old':
#         continue
#     model_path = os.path.join(data_root, data_path, folder, 'models')
#     model_name = 'model_normalized.obj'
#     mtl_name = 'model_normalized.mtl'
#     new_name = folder
#     print(folder)
#     shutil.copy(os.path.join(model_path, model_name), os.path.join(model_path, new_name+'.obj'))
#     shutil.copy(os.path.join(model_path, mtl_name), os.path.join(model_path, new_name+'.mtl'))
#     # exit()
#     #* copy and move to mug folder
#     shutil.copy(os.path.join(model_path, new_name+'.obj'), os.path.join(data_root, 'mug'))
#     shutil.copy(os.path.join(model_path, new_name+'.mtl'), os.path.join(data_root, 'mug'))
#     # exit()
# print('done rename, copy and move')

#* watertight all the object in data_root and simplify it
# category_list = ['mug']
# for category in category_list:
#     data_path = os.path.join(data_root, category)
#     obj_list = [f for f in os.listdir(data_path) if f.endswith('.obj')]
#     obj_list = [f for f in obj_list if not f.endswith('.watertight.obj')]
#     for obj in obj_list:
#         obj_path = os.path.join(data_path, obj)
#         #* watertight to temp.watertight.obj and simplify to original name
#         os.system('Manifold/build/manifold {} {} -s'.format(obj_path, os.path.join(data_path, obj+'.watertight.obj')))
#         os.system('Manifold/build/simplify -i {} -o {} -m -r 0.02'.format(os.path.join(data_path, obj+'.watertight.obj'), obj_path))

# print('done watertight and simplify!!!!')

#* change simplified .obj to .stl
# category_list = ['mug']
# for category in category_list:
#     data_path = os.path.join(data_root, category)
#     obj_list = [f for f in os.listdir(data_path) if f.endswith('.obj')]
#     obj_list = [f for f in obj_list if not f.endswith('.watertight.obj')]
#     for obj in obj_list:
#         obj_path = os.path.join(data_path, obj)
#         stl_path = os.path.join(data_path, obj[:-4]+'.stl')
#         os.system('meshlabserver -i {} -o {}'.format(obj_path, stl_path))

# print('done change .obj to .stl!!!!')

#######################################################*
# preprocess for h5py file
print('start preprocess for grasp file')
def filter_single_grasp(sum_quality, grasps):
    first_grasp_candidate = np.unique(grasps[:, 0, :, :], axis=0)
    second_grasp_candidate = np.unique(grasps[:, 1, :, :], axis=0)
    total_grasp_candidate = np.concatenate((first_grasp_candidate, second_grasp_candidate), axis=0)
    unique_single_grasp_candidate = np.unique(total_grasp_candidate, axis=0)
    
    paired_first_grasp_quality = []
    paired_idx_mapping = {}
    for i in range(len(unique_single_grasp_candidate)):
        first_grasp = total_grasp_candidate[i]
        paired_grasp_idxs = find_paired_grasp(first_grasp, grasps)
        total_quality = 0
        
        paired_idx_list = []
        for idx in paired_grasp_idxs:
            total_quality += sum_quality[idx]
            
            if np.all(first_grasp == grasps[idx][0]):
                paired_idx_list.append([idx, 1])
            else:
                paired_idx_list.append([idx, 0])
            
        total_quality /= len(paired_grasp_idxs)
        paired_first_grasp_quality.append(total_quality)
        paired_idx_mapping[i] = paired_idx_list

    paired_first_grasp_quality = np.array(paired_first_grasp_quality)
        
    return unique_single_grasp_candidate, paired_first_grasp_quality, paired_idx_mapping
        


def find_paired_grasp(first_grasp, bimanual_grasps):
    index_list = []
    grasp_transform = bimanual_grasps

    for i in range(len(grasp_transform)):

        if np.array_equal(first_grasp, grasp_transform[i][0]) or np.array_equal(first_grasp, grasp_transform[i][1]):
            index_list.append(i)
    
    return index_list

grasp_data_root = 'da2_dataset/grasps'
grasp_data_root_processed = 'da2_dataset/grasps_processed'
file_list = os.listdir(grasp_data_root)
files = [os.path.join(grasp_data_root, f) for f in file_list]

flag = 0
for file in tqdm(files):
    grasp_file = h5py.File(file, 'r')
    
    grasps = np.asarray(grasp_file['grasps/transforms'])
    
    force_closure = np.array(grasp_file["/grasps/qualities/Force_closure"])
    torque_optimization = np.array(grasp_file["grasps/qualities/Torque_optimization"])
    dexterity = np.array(grasp_file["grasps/qualities/Dexterity"])
    
    force_closure_weight = 0.4
    torque_optimization_weight = 0.1
    dexterity_weight = 0.5
    
    sum_quality = force_closure_weight * force_closure + torque_optimization_weight * torque_optimization + \
                    dexterity_weight * dexterity
    
    
    # print(sum_quality.shape)
    sum_quality = sum_quality.reshape(-1)
    sum_quality_idx = np.where(sum_quality.reshape(-1) > 0.85)[0]

    if len(sum_quality_idx) == 0:
        print('no grasp quality is over 0.85')
        flag = flag + 1

print(flag)
    # single_grasp, single_grasp_quality, paired_idx_mapping = filter_single_grasp(sum_quality, grasps)
    
    # file_name  = file.split('/')[-1]
    # #copy file to processed folder
    # shutil.copy(file, os.path.join(grasp_data_root_processed, file_name))
    # # add new dataset in grasp group
    # processed_file = h5py.File(os.path.join(grasp_data_root_processed, file_name), 'a')
    # processed_file.create_dataset('grasps/single_grasps', data=single_grasp)
    # processed_file.create_dataset('grasps/single_grasps_quality', data=single_grasp_quality)
    # processed_file.create_dataset('grasps/paired_idx_mapping', data=str(paired_idx_mapping))
    # processed_file.close()
