import torch
import numpy as np
import open3d as o3d
import math
# import mayavi.mlab as mlab
def control_point_l1_loss_better_than_threshold(pred_control_points,
                                                gt_control_points,
                                                confidence,
                                                confidence_threshold,
                                                device="cpu"):
    npoints = pred_control_points.shape[1]
    mask = torch.greater_equal(confidence, confidence_threshold)
    mask_ratio = torch.mean(mask)
    mask = torch.repeat_interleave(mask, npoints, dim=1)
    p1 = pred_control_points[mask]
    p2 = gt_control_points[mask]

    return control_point_l1_loss(p1, p2), mask_ratio


def accuracy_better_than_threshold(pred_success_logits,
                                   gt,
                                   confidence,
                                   confidence_threshold,
                                   device="cpu"):
    """
      Computes average precision for the grasps with confidence > threshold.
    """
    pred_classes = torch.argmax(pred_success_logits, -1)
    correct = torch.equal(pred_classes, gt)
    mask = torch.squeeze(torch.greater_equal(confidence, confidence_threshold),
                         -1)

    positive_acc = torch.sum(correct * mask * gt) / torch.max(
        torch.sum(mask * gt), torch.tensor(1))
    negative_acc = torch.sum(correct * mask * (1. - gt)) / torch.max(
        torch.sum(mask * (1. - gt)), torch.tensor(1))

    return 0.5 * (positive_acc + negative_acc), torch.sum(mask) / gt.shape[0]


def control_point_l1_loss(pred_control_points,
                          gt_control_points,
                          confidence=None,
                          confidence_weight=None,
                          device="cpu",
                          is_bimanual_v2=False,
                          point_loss=False,
                          pred_middle_point=None,
                          use_anchor=False,
                          pc=None,
                          use_block=False,
                          use_angle_loss=False):
    """
      Computes the l1 loss between the predicted control points and the
      groundtruth control points on the gripper.
    """
    #print('control_point_l1_loss', pred_control_points.shape,
    #      gt_control_points.shape)
    angle_error = 0
    
    if not is_bimanual_v2:
        error = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)

        if point_loss:
            gt_middle_point1 = (gt_control_points[:,4] + gt_control_points[:,5]) / 2
            error_point1 = torch.sum(torch.abs(pred_middle_point - gt_middle_point1), -1)
            error = torch.cat((error, error_point1.unsqueeze(-1)), -1)
        
    else:
        if point_loss:
            assert pred_middle_point is not None
            #* vis pred control point and gt control point
            # mlab.figure(bgcolor=(1,1,1))
            # pc = pc.detach().cpu()
            # pc = pc.numpy()
            # pc = pc[0]
            # mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color=(0.1,0.1,1), scale_factor=0.01)
            # for i in range(pred_control_points.shape[1]):
            #     pred_grasp_pc1 = pred_control_points[0,i].detach().cpu().numpy()
            #     mid_point1 = 0.5*(pred_grasp_pc1[2, :] + pred_grasp_pc1[3, :])
            #     modified_pred_grasp1= []
            #     modified_pred_grasp1.append(pred_grasp_pc1[0])
            #     modified_pred_grasp1.append(mid_point1)
            #     modified_pred_grasp1.append(pred_grasp_pc1[2])
            #     modified_pred_grasp1.append(pred_grasp_pc1[4])
            #     modified_pred_grasp1.append(pred_grasp_pc1[2])
            #     modified_pred_grasp1.append(pred_grasp_pc1[3])
            #     modified_pred_grasp1.append(pred_grasp_pc1[5])
            #     pred_grasp_pc1 = np.asarray(modified_pred_grasp1)
                
            #     pred_grasp_pc2 = pred_control_points[1,i].detach().cpu().numpy()
            #     mid_point2 = 0.5*(pred_grasp_pc2[2, :] + pred_grasp_pc2[3, :])
            #     modified_pred_grasp2 = []
            #     modified_pred_grasp2.append(pred_grasp_pc2[0])
            #     modified_pred_grasp2.append(mid_point2)
            #     modified_pred_grasp2.append(pred_grasp_pc2[2])
            #     modified_pred_grasp2.append(pred_grasp_pc2[4])
            #     modified_pred_grasp2.append(pred_grasp_pc2[2])
            #     modified_pred_grasp2.append(pred_grasp_pc2[3])
            #     modified_pred_grasp2.append(pred_grasp_pc2[5])
            #     pred_grasp_pc2 = np.asarray(modified_pred_grasp2)
                
            #     # same way with pred grasp
            #     gt_grasp_pc1 = gt_control_points[0,i].detach().cpu().numpy()
            #     mid_point1 = 0.5*(gt_grasp_pc1[2, :] + gt_grasp_pc1[3, :])
            #     modified_gt_grasp1= []
            #     modified_gt_grasp1.append(gt_grasp_pc1[0])
            #     modified_gt_grasp1.append(mid_point1)
            #     modified_gt_grasp1.append(gt_grasp_pc1[2])
            #     modified_gt_grasp1.append(gt_grasp_pc1[4])
            #     modified_gt_grasp1.append(gt_grasp_pc1[2])
            #     modified_gt_grasp1.append(gt_grasp_pc1[3])
            #     modified_gt_grasp1.append(gt_grasp_pc1[5])
            #     gt_grasp_pc1 = np.asarray(modified_gt_grasp1)
                
            #     gt_grasp_pc2 = gt_control_points[1,i].detach().cpu().numpy()
            #     mid_point2 = 0.5*(gt_grasp_pc2[2, :] + gt_grasp_pc2[3, :])
            #     modified_gt_grasp2 = []
            #     modified_gt_grasp2.append(gt_grasp_pc2[0])
            #     modified_gt_grasp2.append(mid_point2)
            #     modified_gt_grasp2.append(gt_grasp_pc2[2])
            #     modified_gt_grasp2.append(gt_grasp_pc2[4])
            #     modified_gt_grasp2.append(gt_grasp_pc2[2])
            #     modified_gt_grasp2.append(gt_grasp_pc2[3])
            #     modified_gt_grasp2.append(gt_grasp_pc2[5])
            #     gt_grasp_pc2 = np.asarray(modified_gt_grasp2)
                
                
            #     mlab.plot3d(pred_grasp_pc1[:, 0], pred_grasp_pc1[:, 1], pred_grasp_pc1[:, 2], color=(1,0,0), tube_radius=0.003, opacity=1)
            #     mlab.plot3d(pred_grasp_pc2[:, 0], pred_grasp_pc2[:, 1], pred_grasp_pc2[:, 2], color=(1,0,0), tube_radius=0.003, opacity=1)
            #     mlab.plot3d(gt_grasp_pc1[:, 0], gt_grasp_pc1[:, 1], gt_grasp_pc1[:, 2], color=(0,1,0), tube_radius=0.003, opacity=1)
            #     mlab.plot3d(gt_grasp_pc2[:, 0], gt_grasp_pc2[:, 1], gt_grasp_pc2[:, 2], color=(0,1,0), tube_radius=0.003, opacity=1)
                
            # mlab.show()
            # exit()
                
                
            
            
            error1 = torch.sum(torch.abs(pred_control_points[0] - gt_control_points[0]), -1)
            error2 = torch.sum(torch.abs(pred_control_points[1] - gt_control_points[1]), -1)
            gt_middle_point1 = (gt_control_points[0,:,4] + gt_control_points[0,:,5]) / 2
            gt_middle_point2 = (gt_control_points[1,:,4] + gt_control_points[1,:,5]) / 2
            error_point1 = torch.sum(torch.abs(pred_middle_point[0] - gt_middle_point1), -1)
            error_point2 = torch.sum(torch.abs(pred_middle_point[1] - gt_middle_point2), -1)
            error1 = torch.cat((error1, error_point1.unsqueeze(-1)), -1)
            error2 = torch.cat((error2, error_point2.unsqueeze(-1)), -1)
            
            error = error1 + error2
        else:

            error1 = torch.sum(torch.abs(pred_control_points[0] - gt_control_points[0]), -1) # (batch, 6)
            error2 = torch.sum(torch.abs(pred_control_points[1] - gt_control_points[1]), -1)
            error = error1 + error2 # (batch, 6)
            if use_angle_loss:

            # for angle loss between two grasps
                gt_middle_point1 = (gt_control_points[0,:,4] + gt_control_points[0,:,5]) / 2 # (batch, 3)
                gt_middle_point2 = (gt_control_points[1,:,4] + gt_control_points[1,:,5]) / 2
                gt_dot = torch.sum(gt_middle_point1*gt_middle_point2, dim=1) # (batch)
                gt_norm = torch.norm(gt_middle_point1, dim=1) * torch.norm(gt_middle_point2, dim=1) # (batch)
                gt_angle = torch.acos(torch.clamp(gt_dot/gt_norm, -1, 1)) # (batch)
                gt_degrees = gt_angle * (180/math.pi)

                pred_middle_point1 = (pred_control_points[0,:,4] + pred_control_points[0,:,5]) / 2
                pred_middle_point2 = (pred_control_points[1,:,4] + pred_control_points[1,:,5]) / 2
                pred_dot = torch.sum(pred_middle_point1*pred_middle_point2, dim=1)
                pred_norm = torch.norm(pred_middle_point1, dim=1) * torch.norm(pred_middle_point2, dim=1)
                pred_angle = torch.acos(torch.clamp(pred_dot/pred_norm, -1, 1))
                pred_degrees = pred_angle * (180/math.pi)
                
                angle_error = (gt_degrees - pred_degrees) % 360 # Ensure angles are within [0, 360)
                angle_error = torch.min(angle_error, 360-angle_error) # Take the minimum angle difference

                angle_error = torch.mean(torch.pow(angle_error, 2),-1) # Mean squared error 

                
    if not use_anchor:
        error = torch.mean(error, -1)

    if confidence is not None:

        assert (confidence_weight is not None)
        error *= confidence
        confidence_term = torch.mean(
            torch.log(torch.max(
                confidence,
                torch.tensor(1e-10).to(device)))) * confidence_weight
        #print('confidence_term = ', confidence_term.shape)

    #print('l1_error = {}'.format(error.shape))
    if confidence is None:
        return torch.mean(error)
    else:
        return torch.mean(error), -confidence_term, angle_error

def classification_with_confidence_loss(pred_logit,
                                        gt,
                                        confidence,
                                        confidence_weight,
                                        device="cpu"):
    """
      Computes the cross entropy loss and confidence term that penalizes
      outputing zero confidence. Returns cross entropy loss and the confidence
      regularization term.
    """
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logit, gt)
    confidence_term = torch.mean(
        torch.log(torch.max(
            confidence,
            torch.tensor(1e-10).to(device)))) * confidence_weight

    return classification_loss, -confidence_term


def min_distance_loss(pred_control_points,
                      gt_control_points,
                      confidence=None,
                      confidence_weight=None,
                      threshold=None,
                      device="cpu",
                      is_bimanual_v2=False,
                      point_loss=False,
                      pred_middle_point=None):
    """
    Computes the minimum distance (L1 distance)between each gt control point 
    and any of the predicted control points.

    Args: 
      pred_control_points: tensor of (N_pred, M, 4) shape. N is the number of
        grasps. M is the number of points on the gripper.
      gt_control_points: (N_gt, M, 4)
      confidence: tensor of N_pred, tensor for the confidence of each 
        prediction.
      confidence_weight: float, the weight for confidence loss.
    """
    pred_shape = pred_control_points.shape
    gt_shape = gt_control_points.shape

    # if len(pred_shape) != 3:
    #     raise ValueError(
    #         "pred_control_point should have len of 3. {}".format(pred_shape))
    # if len(gt_shape) != 3:
    #     raise ValueError(
    #         "gt_control_point should have len of 3. {}".format(gt_shape))
    # if pred_shape != gt_shape:
    #     raise ValueError("shapes do no match {} != {}".format(
    #         pred_shape, gt_shape))

    # N_pred x Ngt x M x 3
    if not is_bimanual_v2:
        error = pred_control_points.unsqueeze(1) - gt_control_points.unsqueeze(0)
        error = torch.sum(torch.abs(error),
                        -1)  # L1 distance of error (N_pred, N_gt, M)
        error = torch.mean(
            error, -1)  # average L1 for all the control points. (N_pred, N_gt)

        min_distance_error, closest_index = error.min(
            0)  #[0]  # take the min distance for each gt control point. (N_gt)
    else:
        if point_loss:
            assert pred_middle_point is not None
            error1 = pred_control_points[0].unsqueeze(1) - gt_control_points[0].unsqueeze(0) # (N_pred, N_gt, M, 3)
            error2 = pred_control_points[1].unsqueeze(1) - gt_control_points[1].unsqueeze(0) # (N_pred, N_gt, M, 3)
            error1 = torch.sum(torch.abs(error1), -1)  # L1 distance of error (N_pred, N_gt, M)
            error2 = torch.sum(torch.abs(error2), -1)  # L1 distance of error (N_pred, N_gt, M)

            gt_middle_point1 = (gt_control_points[0,:,4] + gt_control_points[0,:,5]) / 2
            gt_middle_point2 = (gt_control_points[1,:,4] + gt_control_points[1,:,5]) / 2
            error_point1 = pred_middle_point[0].unsqueeze(1) - gt_middle_point1.unsqueeze(0)
            error_point2 = pred_middle_point[1].unsqueeze(1) - gt_middle_point2.unsqueeze(0)

            error_point1 = torch.sum(torch.abs(error_point1), -1).unsqueeze(-1)  # L1 distance of error (N_pred, N_gt, M)
            error_point2 = torch.sum(torch.abs(error_point2), -1).unsqueeze(-1)  # L1 distance of error (N_pred, N_gt, M)
            
            error1 = torch.cat((error1, error_point1), -1)
            error2 = torch.cat((error2, error_point2), -1)

            error = error1 + error2
            error = torch.mean(error, -1)
            min_distance_error, closest_index = error.min(0)


        else:
            error1 = pred_control_points[0].unsqueeze(1) - gt_control_points[0].unsqueeze(0) # (N_pred, N_gt, M, 3)
            error2 = pred_control_points[1].unsqueeze(1) - gt_control_points[1].unsqueeze(0) # (N_pred, N_gt, M, 3)
            error1 = torch.sum(torch.abs(error1), -1)  # L1 distance of error (N_pred, N_gt, M)
            error2 = torch.sum(torch.abs(error2), -1)  # L1 distance of error (N_pred, N_gt, M)
            error  = error1 + error2
            error = torch.mean(error, -1)
            min_distance_error, closest_index = error.min(0)


    #print('min_distance_error', get_shape(min_distance_error))
    if confidence is not None:
        #print('closest_index', get_shape(closest_index))
        selected_confidence = torch.nn.functional.one_hot(
            closest_index,
            num_classes=closest_index.shape[0]).float()  # (N_gt, N_pred)
        selected_confidence *= confidence
        #print('selected_confidence', selected_confidence)
        selected_confidence = torch.sum(selected_confidence, -1)  # N_gt
        #print('selected_confidence', selected_confidence)
        min_distance_error *= selected_confidence
        confidence_term = torch.mean(
            torch.log(torch.max(
                confidence,
                torch.tensor(1e-4).to(device)))) * confidence_weight
    else:
        confidence_term = 0.

    return torch.mean(min_distance_error), -confidence_term


def min_distance_better_than_threshold(pred_control_points,
                                       gt_control_points,
                                       confidence,
                                       confidence_threshold,
                                       device="cpu"):
    error = torch.expand_dims(pred_control_points, 1) - torch.expand_dims(
        gt_control_points, 0)
    error = torch.sum(torch.abs(error),
                      -1)  # L1 distance of error (N_pred, N_gt, M)
    error = torch.mean(
        error, -1)  # average L1 for all the control points. (N_pred, N_gt)
    error = torch.min(error, -1)  # (B, N_pred)
    mask = torch.greater_equal(confidence, confidence_threshold)
    mask = torch.squeeze(mask, dim=-1)

    return torch.mean(error[mask]), torch.mean(mask)


def kl_divergence(mu, log_sigma, device="cpu"):
    """
      Computes the kl divergence for batch of mu and log_sigma.
    """
    return torch.mean(
        -.5 * torch.sum(1. + log_sigma - mu**2 - torch.exp(log_sigma), dim=-1))


def confidence_loss(confidence, confidence_weight, device="cpu"):
    return torch.mean(
        torch.log(torch.max(
            confidence,
            torch.tensor(1e-10).to(device)))) * confidence_weight
