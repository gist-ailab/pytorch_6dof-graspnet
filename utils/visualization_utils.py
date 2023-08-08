from __future__ import print_function

import mayavi.mlab as mlab
from utils import utils, sample
import numpy as np
import trimesh
import copy


def get_color_plasma_org(x):
    import matplotlib.pyplot as plt
    return tuple([x for i, x in enumerate(plt.cm.plasma(x)) if i < 3])


def get_color_plasma(x):
    return tuple([float(1 - x), float(x), float(0)])


def plot_mesh(mesh):
    assert type(mesh) == trimesh.base.Trimesh
    mlab.triangular_mesh(mesh.vertices[:, 0],
                         mesh.vertices[:, 1],
                         mesh.vertices[:, 2],
                         mesh.faces,
                         colormap='Blues')


def draw_scene(pc,
               grasps=[],
               grasp_scores=None,
               grasp_color=None,
               gripper_color=(0, 1, 0),
               mesh=None,
               show_gripper_mesh=False,
               grasps_selection=None,
               visualize_diverse_grasps=False,
               min_seperation_distance=0.03,
               pc_color=None,
               plasma_coloring=False,
               target_cps=None,
               is_bimanual=False,
               is_bimanual_v2=False,):
    """
    Draws the 3D scene for the object and the scene.
    Args:
      pc: point cloud of the object
      grasps: list of 4x4 numpy array indicating the transformation of the grasps.
        grasp_scores: grasps will be colored based on the scores. If left 
        empty, grasps are visualized in green.
      grasp_color: if it is a tuple, sets the color for all the grasps. If list
        is provided it is the list of tuple(r,g,b) for each grasp.
      mesh: If not None, shows the mesh of the object. Type should be trimesh 
         mesh.
      show_gripper_mesh: If True, shows the gripper mesh for each grasp. 
      grasp_selection: if provided, filters the grasps based on the value of 
        each selection. 1 means select ith grasp. 0 means exclude the grasp.
      visualize_diverse_grasps: sorts the grasps based on score. Selects the 
        top score grasp to visualize and then choose grasps that are not within
        min_seperation_distance distance of any of the previously selected
        grasps. Only set it to True to declutter the grasps for better
        visualization.
      pc_color: if provided, should be a n x 3 numpy array for color of each 
        point in the point cloud pc. Each number should be between 0 and 1.
      plasma_coloring: If True, sets the plasma colormap for visualizting the 
        pc.
    """

    if is_bimanual or is_bimanual_v2:
        max_grasps=385
    else:
        max_grasps = 100
    grasps = np.array(grasps)

    if grasp_scores is not None:
        grasp_scores = np.array(grasp_scores)

    if len(grasps) > max_grasps:

        print('Downsampling grasps, there are too many')
        chosen_ones = np.random.randint(low=0,
                                        high=len(grasps),
                                        size=max_grasps)
        grasps = grasps[chosen_ones]
        if grasp_scores is not None:
            grasp_scores = grasp_scores[chosen_ones]

    if mesh is not None:
        if type(mesh) == list:
            for elem in mesh:
                plot_mesh(elem)
        else:
            plot_mesh(mesh)

    if pc_color is None and pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc[:, 2],
                          colormap='plasma')
        else:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          color=(0.1, 0.1, 1),
                          scale_factor=0.01)
    elif pc is not None:
        if plasma_coloring:
            mlab.points3d(pc[:, 0],
                          pc[:, 1],
                          pc[:, 2],
                          pc_color[:, 0],
                          colormap='plasma')
        else:
            rgba = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            rgba[:, :3] = np.asarray(pc_color)
            rgba[:, 3] = 255
            src = mlab.pipeline.scalar_scatter(pc[:, 0], pc[:, 1], pc[:, 2])
            src.add_attribute(rgba, 'colors')
            src.data.point_data.set_active_scalars('colors')
            g = mlab.pipeline.glyph(src)
            g.glyph.scale_mode = "data_scaling_off"
            g.glyph.glyph.scale_factor = 0.01

    if is_bimanual or is_bimanual_v2:
        # grasp_pc = np.squeeze(utils.get_control_point_tensor(1, False, is_bimanual=True), 0) #(6,3)
        grasp_pc = np.array([[0,0,-0.03375], [0,0,-0.03375], [0.0425, 0, 0],
                          [-0.0425, 0, 0], [0.0425, 0, 0.0675],
                          [-0.0425, 0, 0.0675]])

        grasp_pc_trans = copy.deepcopy(grasp_pc)
        grasp_pc_trans[:, 2] = -grasp_pc[:, 1]
        grasp_pc_trans[:, 1] = grasp_pc[:, 2]
        grasp_pc = grasp_pc_trans

        mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])
        modified_grasp_pc = []
        modified_grasp_pc.append(grasp_pc[0])
        modified_grasp_pc.append(mid_point)
        modified_grasp_pc.append(grasp_pc[2])
        modified_grasp_pc.append(grasp_pc[4])
        modified_grasp_pc.append(grasp_pc[2])
        modified_grasp_pc.append(grasp_pc[3])
        modified_grasp_pc.append(grasp_pc[5])
        grasp_pc = np.asarray(modified_grasp_pc)

    else:
        grasp_pc = np.squeeze(utils.get_control_point_tensor(1, False), 0)
        grasp_pc[2, 2] = 0.059
        grasp_pc[3, 2] = 0.059

        mid_point = 0.5 * (grasp_pc[2, :] + grasp_pc[3, :])

        modified_grasp_pc = []
        modified_grasp_pc.append(np.zeros((3, ), np.float32))
        modified_grasp_pc.append(mid_point)
        modified_grasp_pc.append(grasp_pc[2])
        modified_grasp_pc.append(grasp_pc[4])
        modified_grasp_pc.append(grasp_pc[2])
        modified_grasp_pc.append(grasp_pc[3])
        modified_grasp_pc.append(grasp_pc[5])

        grasp_pc = np.asarray(modified_grasp_pc)

    def transform_grasp_pc(g):
        output = np.matmul(grasp_pc, g[:3, :3].T)
        output += np.expand_dims(g[:3, 3], 0)

        return output

    if grasp_scores is not None:
        indexes = np.argsort(-np.asarray(grasp_scores))
    else:
        if is_bimanual_v2:
            indexes = range(grasps.shape[1])
        else:
            indexes = range(len(grasps))

    print('draw scene ', len(grasps))

    selected_grasps_so_far = []
    removed = 0

    if grasp_scores is not None:
        min_score = np.min(grasp_scores)
        max_score = np.max(grasp_scores)
        top5 = np.array(grasp_scores).argsort()[-5:][::-1]

    database1 = []
    database2 = []
    marker1 = []
    marker2 = []
    if is_bimanual_v2:
        grasps_length = grasps.shape[1]
    else:
        grasps_length = len(grasps)
    
    for ii in range(grasps_length):
        i = indexes[ii]
        if grasps_selection is not None:
            if grasps_selection[i] == False:
                continue
        if is_bimanual_v2:
            g = grasps[:, i]
        else:
            g = grasps[i]
        is_diverse = True
        for prevg in selected_grasps_so_far:
            distance = np.linalg.norm(prevg[:3, 3] - g[:3, 3])

            if distance < min_seperation_distance:
                is_diverse = False
                break

        if visualize_diverse_grasps:
            if not is_diverse:
                removed += 1
                continue
            else:
                if grasp_scores is not None:
                    print('selected', i, grasp_scores[i], min_score, max_score)
                else:
                    print('selected', i)
                selected_grasps_so_far.append(g)

        if isinstance(gripper_color, list):
            pass
        elif grasp_scores is not None:
            normalized_score = (grasp_scores[i] -
                                min_score) / (max_score - min_score + 0.0001)
            if grasp_color is not None:
                gripper_color = grasp_color[ii]
            else:
                gripper_color = get_color_plasma(normalized_score)

            if min_score == 1.0:
                gripper_color = (0.0, 1.0, 0.0)

        if show_gripper_mesh:
            gripper_mesh = sample.Object(
                'gripper_models/panda_gripper.obj').mesh
            gripper_mesh.apply_transform(g)
            mlab.triangular_mesh(
                gripper_mesh.vertices[:, 0],
                gripper_mesh.vertices[:, 1],
                gripper_mesh.vertices[:, 2],
                gripper_mesh.faces,
                color=gripper_color,
                opacity=1 if visualize_diverse_grasps else 0.5)
        else:
            if is_bimanual_v2:
                g_pred1 = g[0]
                g_pred2 = g[1]
                g_gt1 = g[2]
                g_gt2 = g[3]
                
                # transform grasp pc according to the predicted grasp and gt grasp
                pts_pred1 = np.matmul(grasp_pc, g_pred1[:3, :3].T)
                pts_pred1 += np.expand_dims(g_pred1[:3, 3], 0)
                pts_pred2 = np.matmul(grasp_pc, g_pred2[:3, :3].T)
                pts_pred2 += np.expand_dims(g_pred2[:3, 3], 0)
                pts_gt1 = np.matmul(grasp_pc, g_gt1[:3, :3].T)
                pts_gt1 += np.expand_dims(g_gt1[:3, 3], 0)
                pts_gt2 = np.matmul(grasp_pc, g_gt2[:3, :3].T)
                pts_gt2 += np.expand_dims(g_gt2[:3, 3], 0)
                
                current_pred_t1 = countX(database1, g_pred1)
                current_pred_t2 = countX(database1, g_pred2)
                # add sphere end of the predicted gripper
                tmp_pred1 = np.asarray([0,-0.03375-0.02*current_pred_t1,0]).reshape(-1,3)
                trans_pred1 = np.matmul(tmp_pred1, g_pred1[:3, :3].T)
                trans_pred1 += np.expand_dims(g_pred1[:3, 3], 0)
                tmp_pred2 = np.asarray([0,-0.03375-0.02*current_pred_t2,0]).reshape(-1,3)
                trans_pred2 = np.matmul(tmp_pred2, g_pred2[:3, :3].T)
                trans_pred2 += np.expand_dims(g_pred2[:3, 3], 0)
                if (i*5)/grasps.shape[1] > 1:
                    color = (i*5)/grasps.shape[1] - (i*5)//grasps.shape[1]
                else:
                    color = (i*5)/grasps.shape[1]
                marker_color_pred = (color, 0, color)
                
                
                current_gt_t1 = countX(database2, g_gt1)
                current_gt_t2 = countX(database2, g_gt2)
                # add sphere end of the ground truth gripper
                tmp_gt1 = np.asarray([0,-0.03375-0.02*current_gt_t1,0]).reshape(-1,3)
                trans_gt1 = np.matmul(tmp_gt1, g_gt1[:3, :3].T)
                trans_gt1 += np.expand_dims(g_gt1[:3, 3], 0)
                tmp_gt2 = np.asarray([0,-0.03375-0.02*current_gt_t2,0]).reshape(-1,3)
                trans_gt2 = np.matmul(tmp_gt2, g_gt2[:3, :3].T)
                trans_gt2 += np.expand_dims(g_gt2[:3, 3], 0)
                marker_color_gt = (color, 0, color)
                # add grasp to database for counting
                database1.append(g_pred1)
                database1.append(g_pred1)
                database2.append(g_gt1)
                database2.append(g_gt2)
                
                # plot the predicted gripper and gt gripper and their end spheres
                mlab.plot3d(pts_pred1[:, 0],
                            pts_pred1[:, 1],
                            pts_pred1[:, 2],
                            color=(1.0, 0.0, 0.0),
                            tube_radius=0.003,
                            opacity=1)
                mlab.plot3d(pts_pred2[:, 0],
                            pts_pred2[:, 1],
                            pts_pred2[:, 2],
                            color=(1.0, 0.0, 0.0),
                            tube_radius=0.003,
                            opacity=1)
                mlab.plot3d(pts_gt1[:, 0],
                            pts_gt1[:, 1],
                            pts_gt1[:, 2],
                            color=(0.0, 1.0, 0.0),
                            tube_radius=0.003,
                            opacity=1)
                mlab.plot3d(pts_gt2[:, 0],
                            pts_gt2[:, 1],
                            pts_gt2[:, 2],
                            color=(0.0, 1.0, 0.0),
                            tube_radius=0.003,
                            opacity=1)
                
                mlab.points3d(trans_gt1[:, 0],
                            trans_gt1[:, 1],
                            trans_gt1[:, 2],
                            color=marker_color_gt,
                            scale_factor=0.02)
                mlab.points3d(trans_gt2[:, 0],
                              trans_gt2[:, 1],
                              trans_gt2[:, 2],
                              color=marker_color_gt,
                              scale_factor=0.02)
                mlab.points3d(trans_pred1[:, 0],
                              trans_pred1[:, 1],
                              trans_pred1[:, 2],
                              color=marker_color_pred,
                              scale_factor=0.02)
                mlab.points3d(trans_pred2[:, 0], 
                              trans_pred2[:, 1],
                              trans_pred2[:, 2],
                              color=marker_color_pred,
                              scale_factor=0.02)
                
            else:
                pts = np.matmul(grasp_pc, g[:3, :3].T)
                pts += np.expand_dims(g[:3, 3], 0)

                if isinstance(gripper_color, list):
                    mlab.plot3d(pts[:, 0],
                                pts[:, 1],
                                pts[:, 2],
                                color=gripper_color[i],
                                tube_radius=0.003,
                                opacity=1)
                else:
                    tube_radius = 0.001
                    mlab.plot3d(pts[:, 0],
                                pts[:, 1],
                                pts[:, 2],
                                color=gripper_color,
                                tube_radius=tube_radius,
                                opacity=1)
                    if target_cps is not None:
                        mlab.points3d(target_cps[ii, :, 0],
                                    target_cps[ii, :, 1],
                                    target_cps[ii, :, 2],
                                    color=(1.0, 0.0, 0),
                                    scale_factor=0.01)

    print('removed {} similar grasps'.format(removed))

def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x).all():
            count = count + 1
    return count


def get_axis():
    # hacky axis for mayavi
    axis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axis_x = np.array([np.linspace(0, 0.10, 50), np.zeros(50), np.zeros(50)]).T
    axis_y = np.array([np.zeros(50), np.linspace(0, 0.10, 50), np.zeros(50)]).T
    axis_z = np.array([np.zeros(50), np.zeros(50), np.linspace(0, 0.10, 50)]).T
    axis = np.concatenate([axis_x, axis_y, axis_z], axis=0)
    return axis
