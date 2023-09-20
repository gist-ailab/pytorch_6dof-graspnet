import torch
from . import networks
from os.path import join
import utils.utils as utils
import math

class GraspNetModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> sampling / evaluation)
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        if self.gpu_ids and self.gpu_ids[0] >= torch.cuda.device_count():
            self.gpu_ids[0] = torch.cuda.device_count() - 1
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.loss = None
        self.pcs = None
        self.grasps = None
        # load/define networks
        self.net = networks.define_classifier(opt, self.gpu_ids, opt.arch,
                                              opt.init_type, opt.init_gain,
                                              self.device)

        self.criterion = networks.define_loss(opt)

        self.confidence_loss = None
        if self.opt.arch == "vae":
            self.kl_loss = None
            self.reconstruction_loss = None
            self.angle_loss = None
        elif self.opt.arch == "gan":
            self.reconstruction_loss = None
        else:
            self.classification_loss = None

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            
            self.total_epoch = self.opt.niter + self.opt.niter_decay
            self.train_data_length = self.opt.train_data_length
            self.total_iter = (self.train_data_length // self.opt.num_objects_per_batch) * self.total_epoch
            self.annealing_agent = Annealer(self.total_iter, shape='cosine')
            
        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch, self.is_train)
            

    def set_input(self, data):
        if not self.opt.is_bimanual_v2:
            if self.opt.use_anchor:
                input_pcs = torch.from_numpy(data['pc']).contiguous()
                input_grasps = torch.from_numpy(data['anchor1']).float()
                if self.opt.arch == "evaluator":
                    targets = torch.from_numpy(data['labels']).float()
                else:
                    targets = torch.from_numpy(data['anchor1']).float()
                self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)

                self.grasps = input_grasps.to(self.device).requires_grad_(
                    self.is_train)
                self.targets = targets.to(self.device)
            
            elif self.opt.use_block:
                input_pcs = torch.from_numpy(data['pc']).contiguous()

                input_grasps = torch.from_numpy(data['grasp_rt']).float().transpose(0, 1)
                features = torch.from_numpy(data['features']).float().transpose(0,1)
                if self.opt.arch == "evaluator":
                    targets = torch.from_numpy(data['labels']).float()
                else:
                    targets = torch.from_numpy(data['target_cps']).float()
                self.features = features.to(self.device).requires_grad_(self.is_train)
                self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)

                self.grasps = input_grasps.to(self.device).requires_grad_(self.is_train)
                self.targets = targets.to(self.device)          
            
            elif self.opt.second_grasp_sample:
                input_pcs = torch.from_numpy(data['pc']).contiguous()
                input_grasps = torch.from_numpy(data['grasp_rt']).float()
                input_grasps_pair = torch.from_numpy(data['grasp_rt_paired']).float()
                input_grasps = torch.cat([input_grasps, input_grasps_pair], dim=1)
                if self.opt.arch == "evaluator":
                    targets = torch.from_numpy(data['labels']).float()
                else:
                    targets = torch.from_numpy(data['target_cps']).float()
                    targets_pair = torch.from_numpy(data['target_cps_paired']).float()
                    targets = torch.cat([targets.unsqueeze(0), targets_pair.unsqueeze(0)], dim=0)
                
                self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
                self.grasps = input_grasps.to(self.device).requires_grad_(self.is_train)
                self.targets = targets.to(self.device)
            else:
                input_pcs = torch.from_numpy(data['pc']).contiguous()
                input_grasps = torch.from_numpy(data['grasp_rt']).float()
                if self.opt.arch == "evaluator":
                    targets = torch.from_numpy(data['labels']).float()
                else:
                    targets = torch.from_numpy(data['target_cps']).float()

                self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
                self.grasps = input_grasps.to(self.device).requires_grad_(self.is_train)
                self.targets = targets.to(self.device)
        
        else:
            input_pcs = torch.from_numpy(data['pc']).contiguous()
            input_grasps1 = torch.from_numpy(data['grasp_rt1']).float()
            input_grasps2 = torch.from_numpy(data['grasp_rt2']).float()
            input_grasps = torch.cat([input_grasps1, input_grasps2], dim=1)

            if self.opt.arch == "evaluator":
                targets = torch.from_numpy(data['labels']).float()
            else:
                targets1 = torch.from_numpy(data['target_cps1']).float()
                targets2 = torch.from_numpy(data['target_cps2']).float()
                targets = torch.cat([targets1.unsqueeze(0), targets2.unsqueeze(0)], dim=0)

            self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
            self.grasps = input_grasps.to(self.device).requires_grad_(self.is_train)
            self.targets = targets.to(self.device)
            

    def generate_grasps(self, pcs, z=None):
        with torch.no_grad():
            return self.net.module.generate_grasps(pcs, z=z)

    def evaluate_grasps(self, pcs, gripper_pcs):
        success, _ = self.net.module(pcs, gripper_pcs)
        return torch.sigmoid(success)

    def forward(self):
        if self.opt.use_block:
            return self.net(self.pcs, self.grasps, train=self.is_train, features=self.features, targets=self.targets)
        else:
            return self.net(self.pcs, self.grasps, train=self.is_train)
            
    def backward(self, out):
        if self.opt.arch == 'vae':
            if not self.opt.is_bimanual_v2:
                if self.opt.use_anchor:
                    predicted_anchor, confidence, mu, logvar = out
                    self.reconstruction_loss, self.confidence_loss = self.criterion[1](
                        predicted_anchor,
                        self.targets,
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device,
                        use_anchor=self.opt.use_anchor
                    )
                elif self.opt.is_bimanual:
                    if self.opt.use_block:
                        dir1_list, app1_list, point1_list, confidence_list, mu_list, logvar_list, target_list = out
                        self.loss = []
                        self.reconstruction_loss = []
                        self.confidence_loss = []
                        self.kl_loss = []

                        dir1_list = dir1_list.transpose(0, 1)
                        app1_list = app1_list.transpose(0, 1)
                        point1_list = point1_list.transpose(0, 1)
                        confidence_list = confidence_list.transpose(0, 1)
                        mu_list = mu_list.transpose(0, 1)
                        logvar_list = logvar_list.transpose(0, 1)
                        target_list = target_list.transpose(0, 1)
                        for block_idx in range(dir1_list.shape[0]):
                        
                            predicted_cp = utils.transform_control_points_v3(app1_list[block_idx], dir1_list[block_idx], 
                                                                                point1_list[block_idx], app1_list.shape[1], 
                                                                                device=self.device)[:,:,:3]
                            # print('predicted_cp', predicted_cp.shape)
                            
                            reconstruction_loss, confidence_loss = self.criterion[1](
                                predicted_cp,
                                target_list[block_idx],
                                confidence=confidence_list[block_idx],
                                confidence_weight=self.opt.confidence_weight,
                                device=self.device,
                                point_loss=self.opt.use_point_loss,
                                pred_middle_point=point1_list[block_idx],
                                use_block=True
                            )
                            kl_loss = self.opt.kl_loss_weight * self.criterion[0](mu_list[block_idx], logvar_list[block_idx], 
                                                                                        device=self.device)
                            self.reconstruction_loss.append(reconstruction_loss)
                            self.confidence_loss.append(confidence_loss)
                            self.kl_loss.append(kl_loss)
                            self.loss.append(kl_loss + reconstruction_loss + confidence_loss)
                        
                            
                    else:
                        dir1, app1, point1, confidence, mu, logvar = out
                        predicted_cp = utils.transform_control_points_v3(app1, dir1, point1, app1.shape[0], device=self.device)[:,:,:3]
                        self.reconstruction_loss, self.confidence_loss = self.criterion[1](
                            predicted_cp,
                            self.targets,
                            confidence=confidence,
                            confidence_weight=self.opt.confidence_weight,
                            device=self.device,
                            point_loss=self.opt.use_point_loss,
                            pred_middle_point=point1,
                        )
                else:
                    predicted_cp, confidence, mu, logvar = out

                    predicted_cp = utils.transform_control_points(
                        predicted_cp, predicted_cp.shape[0], device=self.device, is_bimanual=self.opt.is_bimanual)[:,:,:3]# (64, 6, 3)

                    self.reconstruction_loss, self.confidence_loss = self.criterion[1](
                        predicted_cp,
                        self.targets,
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device)    
                
            else:
                if self.opt.is_bimanual_v3 and not self.opt.cross_condition:
                    dir1, dir2, app1, app2, point1, point2, confidence, mu, logvar = out

                    predicted_cp1 = utils.transform_control_points_v3(app1, dir1, point1, app1.shape[0], device=self.device)[:,:,:3]
                    predicted_cp2 = utils.transform_control_points_v3(app2, dir2, point2, app2.shape[0], device=self.device)[:,:,:3]
                    predicted_point = torch.cat([point1.unsqueeze(0), point2.unsqueeze(0)], dim=0)
                    predicted_point = predicted_point.to(device=self.device)
                elif self.opt.cross_condition:
                    out1, out2 = out
                    dir11, dir12, app11, app12, point11, point12, confidence1, mu, logvar = out1
                    dir21, dir22, app21, app22, point21, point22, confidence21, confidence22, mu1, logvar1 = out2
                    
                    predicted_cp11 = utils.transform_control_points_v3(app11, dir11, point11, app11.shape[0], device=self.device)[:,:,:3]
                    predicted_cp12 = utils.transform_control_points_v3(app12, dir12, point12, app12.shape[0], device=self.device)[:,:,:3]
                    predicted_point1 = torch.cat([point11.unsqueeze(0), point12.unsqueeze(0)], dim=0)
                    predicted_point1 = predicted_point1.to(device=self.device)
                    
                    predicted_cp21 = utils.transform_control_points_v3(app21, dir21, point21, app21.shape[0], device=self.device)[:,:,:3]
                    predicted_cp22 = utils.transform_control_points_v3(app22, dir22, point22, app22.shape[0], device=self.device)[:,:,:3]
                    predicted_point2 = torch.cat([point21.unsqueeze(0), point22.unsqueeze(0)], dim=0)
                    predicted_point2 = predicted_point2.to(device=self.device)
                else:
                    predicted_cp, confidence, mu, logvar = out
                    if len(self.opt.gpu_ids) > 1:
                        predicted_cp = torch.transpose(predicted_cp, 0, 1)
                        
                    predicted_cp1 = utils.transform_control_points(
                        predicted_cp[0], predicted_cp[0].shape[0], device=self.device, is_bimanual=True)[:,:,:3]
                    predicted_cp2 = utils.transform_control_points(
                        predicted_cp[1], predicted_cp[1].shape[0], device=self.device, is_bimanual=True)[:,:,:3]
                    predicted_point = None
                if not self.opt.cross_condition:
                    predicted_cp = torch.cat([predicted_cp1.unsqueeze(0), predicted_cp2.unsqueeze(0)], dim=0)# (2, 64, 6, 3)
                    predicted_cp = predicted_cp.to(device=self.device)
                    self.reconstruction_loss, self.confidence_loss, self.angle_loss = self.criterion[1](
                        predicted_cp,
                        self.targets,
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device,
                        is_bimanual_v2=self.opt.is_bimanual_v2,
                        point_loss=self.opt.use_point_loss,
                        pred_middle_point=predicted_point,
                        pc=self.pcs,
                        use_angle_loss=self.opt.use_angle_loss)
                else:
                    predicted_cp = torch.cat([predicted_cp11.unsqueeze(0), predicted_cp12.unsqueeze(0)], dim=0)
                    predicted_cp = predicted_cp.to(device=self.device)
                    # predicted_cp2 = torch.cat([predicted_cp21.unsqueeze(0), predicted_cp22.unsqueeze(0)], dim=0)
                    # predicted_cp2 = predicted_cp2.to(device=self.device)

                    reconstruction_loss1, confidence_loss1, _ = self.criterion[1](
                        predicted_cp,
                        self.targets,
                        confidence=confidence1,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device,
                        is_bimanual_v2=self.opt.is_bimanual_v2)
                    reconstruction_loss21, confidence_loss21, _ = self.criterion[1](
                        predicted_cp21,
                        predicted_cp11,
                        confidence=confidence21,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device)
                    reconstruction_loss22, confidence_loss22, _ = self.criterion[1](
                        predicted_cp22,
                        predicted_cp12,
                        confidence=confidence22,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device)

                    self.reconstruction_loss = [reconstruction_loss1, reconstruction_loss21, reconstruction_loss22]
                    self.confidence_loss = [confidence_loss1, confidence_loss21, confidence_loss22]
                    
                    kld1 = self.criterion[0](mu, logvar, device=self.device) * self.opt.kl_loss_weight
                    # kld1 = self.annealing_agent(kld1)
                    kld2 = self.criterion[0](mu1, logvar1, device=self.device) * self.opt.kl_loss_weight
                    # kld2 = self.annealing_agent(kld2)
                    
                    self.kl_loss = [kld1, kld2]
                    self.loss = [reconstruction_loss1 + confidence_loss1 + kld1, 
                                 reconstruction_loss21 + confidence_loss21 + kld2, reconstruction_loss22 + confidence_loss22 + kld2]
                    
            if not self.opt.use_block and not self.opt.cross_condition:
                kld = self.criterion[0](mu, logvar, device=self.device)
                self.kl_loss = self.annealing_agent(kld)
                # self.kl_loss = self.opt.kl_loss_weight * self.criterion[0](
                #         mu, logvar, device=self.device)

                self.loss = self.kl_loss + self.reconstruction_loss + self.confidence_loss + self.angle_loss
                
        elif self.opt.arch == 'gan':
            if self.opt.is_bimanual_v2:
                if self.opt.is_bimanual_v3:
                    dir1, dir2, app1, app2, point1, point2, confidence = out
                    
                    predicted_cp1 = utils.transform_control_points_v3(app1, dir1, point1, app1.shape[0], device=self.device)[:,:,:3]
                    predicted_cp2 = utils.transform_control_points_v3(app2, dir2, point2, app2.shape[0], device=self.device)[:,:,:3]
                    predicted_point = torch.cat([point1.unsqueeze(0), point2.unsqueeze(0)], dim=0)
                    predicted_point = predicted_point.to(device=self.device)
                
                predicted_cp = torch.cat([predicted_cp1.unsqueeze(0), predicted_cp2.unsqueeze(0)], dim=0)# (2, 64, 6, 3)
                predicted_cp = predicted_cp.to(device=self.device)
                
                self.reconstruction_loss, self.confidence_loss = self.criterion(
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device,
                    is_bimanual_v2=self.opt.is_bimanual_v2,
                    point_loss=self.opt.use_point_loss,
                    pred_middle_point=predicted_point)
            
            else:
                predicted_cp, confidence = out
                predicted_cp = utils.transform_control_points(
                    predicted_cp, predicted_cp.shape[0], device=self.device, is_bimanual=self.opt.is_bimanual)[:,:,:3]
            
                self.reconstruction_loss, self.confidence_loss = self.criterion(
                    predicted_cp,
                    self.targets,
                    confidence=confidence,
                    confidence_weight=self.opt.confidence_weight,
                    device=self.device)
                
            self.loss = self.reconstruction_loss + self.confidence_loss
        elif self.opt.arch == 'evaluator':
            grasp_classification, confidence = out
            self.classification_loss, self.confidence_loss = self.criterion(
                grasp_classification.squeeze(),
                self.targets,
                confidence,
                self.opt.confidence_weight,
                device=self.device)
            self.loss = self.classification_loss + self.confidence_loss
        
        if self.opt.use_block:
            for block_idx in range(len(self.loss)-1):
                self.loss[block_idx].backward(retain_graph=True)
            self.loss[-1].backward()
            
            self.reconstruction_loss = torch.stack(self.reconstruction_loss, dim=0)
            self.confidence_loss = torch.stack(self.confidence_loss, dim=0)
            self.kl_loss = torch.stack(self.kl_loss, dim=0)
            self.loss = torch.stack(self.loss, dim=0)
            # print(self.reconstruction_loss.shape)
            
            # self.reconstruction_loss = self.reconstruction_loss.transpose(0, 1)
            # self.confidence_loss = self.confidence_loss.transpose(0, 1)
            # self.kl_loss = self.kl_loss.transpose(0, 1)
            # self.loss = self.loss.transpose(0, 1)
            # exit()
        elif self.opt.cross_condition:
            self.loss[2].backward(retain_graph=True)
            self.loss[1].backward(retain_graph=True)
            self.loss[0].backward()
        else:
            self.loss.backward()

        
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()
        self.annealing_agent.step()
        


##################

    def load_network(self, which_epoch, train=True):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        checkpoint = torch.load(load_path, map_location=self.device)
        if hasattr(checkpoint['model_state_dict'], '_metadata'):
            del checkpoint['model_state_dict']._metadata
        net.load_state_dict(checkpoint['model_state_dict'])
        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.opt.epoch_count = checkpoint["epoch"]
            self.annealing_agent = Annealer(self.total_iter, shape='cosine', current_step=self.opt.epoch_count * (self.train_data_length // self.opt.num_objects_per_batch))
        else:
            net.eval()

    def save_network(self, net_name, epoch_num):
        """save model to disk"""
        save_filename = '%s_net.pth' % (net_name)
        save_path = join(self.save_dir, save_filename)
        torch.save(
            {
                'epoch': epoch_num + 1,
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self, vis=False):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            if self.opt.is_bimanual_v3 and not self.opt.cross_condition:
                dir1, dir2, app1, app2, point1, point2, confidence = out
            elif self.opt.cross_condition:
                out1, out2 = out
            elif self.opt.is_bimanual:
                if self.opt.use_block:
                    dir1, app1, point1, confidence, target_list = out
                else: 
                    dir1, app1, point1, confidence = out
            else:
                prediction, confidence = out
            if vis:
                if self.opt.arch == 'vae':
                    predicted_cp = utils.transform_control_points(
                        prediction, prediction.shape[0], device=self.device, is_bimanual=self.opt.is_bimanual)
                    return predicted_cp[:,:,:3], self.targets, confidence
                
            else:    
                if self.opt.arch == "vae":
                    if not self.opt.is_bimanual_v2:
                        if self.opt.use_anchor:
                            reconstruction_loss, _ = self.criterion[1](
                                prediction,
                                self.targets,
                                confidence=confidence,
                                confidence_weight=self.opt.confidence_weight,
                                device=self.device,
                                use_anchor=self.opt.use_anchor)
                        elif self.opt.is_bimanual:
                            if self.opt.use_block:
                                # self.loss = []
                                # self.reconstruction_loss = []
                                # self.confidence_loss = []
                                # self.kl_loss = []
                                dir1 = dir1.transpose(0, 1)
                                app1 = app1.transpose(0, 1)
                                point1 = point1.transpose(0, 1)
                                confidence = confidence.transpose(0, 1)
                                target_list = target_list.transpose(0, 1)
                                
                                reconstruction_loss_list = []
                                for block_idx in range(dir1.shape[0]):
                                    predicted_cp = utils.transform_control_points_v3(app1[block_idx], dir1[block_idx], 
                                                                                    point1[block_idx], app1[block_idx].shape[0], 
                                                                                    device=self.device)[:,:,:3]
                                    reconstruction_loss, _ = self.criterion[1](
                                        predicted_cp,
                                        target_list[block_idx],
                                        confidence=confidence[block_idx],
                                        confidence_weight=self.opt.confidence_weight,
                                        device=self.device,
                                        point_loss=self.opt.use_point_loss,
                                        pred_middle_point=point1[block_idx],
                                    )
                                    reconstruction_loss_list.append(reconstruction_loss)
                                
                                reconstruction_loss = reconstruction_loss_list
                            else:
                                predicted_cp = utils.transform_control_points_v3(app1, dir1, point1, app1.shape[0], device=self.device)[:,:,:3]
                                reconstruction_loss, _ = self.criterion[1](
                                    predicted_cp,
                                    self.targets,
                                    confidence=confidence,
                                    confidence_weight=self.opt.confidence_weight,
                                    point_loss=self.opt.use_point_loss,
                                    pred_middle_point=point1,
                                )
                        else:
                            predicted_cp = utils.transform_control_points(
                                prediction, prediction.shape[0], device=self.device, is_bimanual=self.opt.is_bimanual)[:,:,:3]
                            reconstruction_loss, _ = self.criterion[1](
                                predicted_cp,
                                self.targets,
                                confidence=confidence,
                                confidence_weight=self.opt.confidence_weight,
                                device=self.device)
                            
                        return reconstruction_loss, 1
                    else:
                        if self.opt.is_bimanual_v3 and not self.opt.cross_condition:
                    
                            predicted_cp1 = utils.transform_control_points_v3(app1, dir1, point1, app1.shape[0], device=self.device)[:,:,:3]
                            predicted_cp2 = utils.transform_control_points_v3(app2, dir2, point2, app2.shape[0], device=self.device)[:,:,:3]
                            predicted_point = torch.cat([point1.unsqueeze(0), point2.unsqueeze(0)], dim=0)
                            predicted_point = predicted_point.to(device=self.device)
                        elif self.opt.cross_condition:
                            dir11, dir12, app11, app12, point11, point12, confidence = out1
                            dir21, dir22, app21, app22, point21, point22, confidence21, confidence22 = out2
                            
                            predicted_cp11 = utils.transform_control_points_v3(app11, dir11, point11, app11.shape[0], device=self.device)[:,:,:3]
                            predicted_cp12 = utils.transform_control_points_v3(app12, dir12, point12, app12.shape[0], device=self.device)[:,:,:3]
                            predicted_point1 = torch.cat([point11.unsqueeze(0), point12.unsqueeze(0)], dim=0)
                            predicted_point1 = predicted_point1.to(device=self.device)
                        
                            predicted_cp21 = utils.transform_control_points_v3(app21, dir21, point21, app21.shape[0], device=self.device)[:,:,:3]
                            predicted_cp22 = utils.transform_control_points_v3(app22, dir22, point22, app22.shape[0], device=self.device)[:,:,:3]
                            predicted_point2 = torch.cat([point21.unsqueeze(0), point22.unsqueeze(0)], dim=0)
                            predicted_point2 = predicted_point2.to(device=self.device)
                            
                        else:
                            if len(self.opt.gpu_ids) > 1:
                                prediction = torch.transpose(prediction, 0, 1)
                            
                            predicted_cp1 = utils.transform_control_points(
                            prediction[0], prediction[0].shape[0], device=self.device, is_bimanual=True)[:,:,:3]
                            predicted_cp2 = utils.transform_control_points(
                                prediction[1], prediction[1].shape[0], device=self.device, is_bimanual=True)[:,:,:3]
                            predicted_point = None
                        if not self.opt.cross_condition:
                            predicted_cp = torch.cat([predicted_cp1.unsqueeze(0), predicted_cp2.unsqueeze(0)], dim=0)# (2, 64, 6, 3)
                            predicted_cp = predicted_cp.to(device=self.device)
                            
                            reconstruction_loss, _, _ = self.criterion[1](
                                predicted_cp,
                                self.targets,
                                confidence=confidence,
                                confidence_weight=self.opt.confidence_weight,
                                device=self.device,
                                is_bimanual_v2=self.opt.is_bimanual_v2,
                                point_loss=self.opt.use_point_loss,
                                pred_middle_point=predicted_point,
                                use_angle_loss=self.opt.use_angle_loss)
                            
                            return reconstruction_loss, 1
                        else:
                            predicted_cp = torch.cat([predicted_cp11.unsqueeze(0), predicted_cp12.unsqueeze(0)], dim=0)
                            predicted_cp = predicted_cp.to(device=self.device)
                            # predicted_cp2 = torch.cat([predicted_cp21.unsqueeze(0), predicted_cp22.unsqueeze(0)], dim=0)
                            # predicted_cp2 = predicted_cp2.to(device=self.device)
                            
                            reconstruction_loss1, _, _ = self.criterion[1](
                                predicted_cp,
                                self.targets,
                                confidence=confidence,
                                confidence_weight=self.opt.confidence_weight,
                                device=self.device,
                                is_bimanual_v2=self.opt.is_bimanual_v2)
                            reconstruction_loss21, _, _ = self.criterion[1](
                                predicted_cp21,
                                predicted_cp11,
                                confidence=confidence21,
                                confidence_weight=self.opt.confidence_weight,
                                device=self.device)
                            reconstruction_loss22, _, _ = self.criterion[1](
                                predicted_cp22,
                                predicted_cp12,
                                confidence=confidence21,
                                confidence_weight=self.opt.confidence_weight,
                                device=self.device)
                            
                            return [reconstruction_loss1, reconstruction_loss21, reconstruction_loss22], 1
                                    
                elif self.opt.arch == "gan":
                    if not self.opt.is_bimanual_v2:
                        predicted_cp = utils.transform_control_points(
                            prediction, prediction.shape[0], device=self.device, is_bimanual=self.opt.is_bimanual)[:,:,:3]
                        reconstruction_loss, _ = self.criterion(
                            predicted_cp,
                            self.targets,
                            confidence=confidence,
                            confidence_weight=self.opt.confidence_weight,
                            device=self.device)
                    else:
                        if self.opt.is_bimanual_v3:
                            predicted_cp1 = utils.transform_control_points_v3(app1, dir1, point1, app1.shape[0], device=self.device)[:,:,:3]
                            predicted_cp2 = utils.transform_control_points_v3(app2, dir2, point2, app2.shape[0], device=self.device)[:,:,:3]
                            predicted_cp = torch.cat([predicted_cp1.unsqueeze(0), predicted_cp2.unsqueeze(0)], dim=0)# (2, 64, 6, 3)
                            predicted_cp = predicted_cp.to(device=self.device)
                            predicted_point = torch.cat([point1.unsqueeze(0), point2.unsqueeze(0)], dim=0)
                            predicted_point = predicted_point.to(device=self.device)

                            reconstruction_loss, _ = self.criterion(
                            predicted_cp,
                            self.targets,
                            confidence=confidence,
                            confidence_weight=self.opt.confidence_weight,
                            device=self.device,
                            is_bimanual_v2=self.opt.is_bimanual_v2,
                            point_loss=self.opt.use_point_loss,
                            pred_middle_point=predicted_point)
                    
                    return reconstruction_loss, 1

                else:
                    predicted = torch.round(torch.sigmoid(prediction)).squeeze()
                    correct = (predicted == self.targets).sum().item()
                    return correct, len(self.targets)

class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    Parameters:
        total_steps (int): Number of epochs to reach full KL divergence weight.
        shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
    """

    def __init__(self, total_steps: int, shape: str, disable=False, current_step=None):
        self.total_steps = total_steps
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step = 1
            
        if not disable:
            self.shape = shape
        else:
            self.shape = 'none'

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == 'linear':
            slope = (self.current_step / self.total_steps)
        elif self.shape == 'cosine':
            slope = 0.5 + 0.5 * math.cos(math.pi * (self.current_step / self.total_steps - 1))
        elif self.shape == 'logistic':
            smoothness = self.total_steps / 10
            exponent = ((self.total_steps / 2) - self.current_step) / smoothness
            slope = 1 / (1 + math.exp(exponent))
        elif self.shape == 'none':
            slope = 1.0
        else:
            raise ValueError('Invalid shape for annealing function. Must be linear, cosine, or logistic.')
        return slope

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        else:
            pass