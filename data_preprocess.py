import numpy as np
import os
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
category_list = ['mug']
for category in category_list:
    data_path = os.path.join(data_root, category)
    obj_list = [f for f in os.listdir(data_path) if f.endswith('.obj')]
    obj_list = [f for f in obj_list if not f.endswith('.watertight.obj')]
    for obj in obj_list:
        obj_path = os.path.join(data_path, obj)
        stl_path = os.path.join(data_path, obj[:-4]+'.stl')
        os.system('meshlabserver -i {} -o {}'.format(obj_path, stl_path))

print('done change .obj to .stl!!!!')
    