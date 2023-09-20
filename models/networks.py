import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from models import losses
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import pointnet2_ops.pointnet2_modules as pointnet2
from time import time
import copy
import utils.utils as utils

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(opt, gpu_ids, arch, init_type, init_gain, device):
    net = None
    if arch == 'vae':
        # if opt.use_block:
        #     net = GraspSamplerVAEBlock(opt.model_scale, opt.pointnet_radius,
        #                                 opt.pointnet_nclusters, opt.latent_size, device)
        if opt.use_anchor:
            net = GraspSamplerVAEAnchor(opt.model_scale, opt.pointnet_radius,
                                        opt.pointnet_nclusters, opt.latent_size, device)
        elif opt.use_block:
            net = GraspSamplerVAEBlock(opt.model_scale, opt.pointnet_radius, opt.pointnet_nclusters, opt.latent_size, device)
        elif opt.cross_condition:
            net = GraspSamplerCrossConditionVAE(opt.model_scale, opt.pointnet_radius, opt.pointnet_nclusters, 
                                                opt.latent_size, device, cross_condition=opt.cross_condition)
        else:
            net = GraspSamplerVAE(opt.model_scale, opt.pointnet_radius,
                                opt.pointnet_nclusters, opt.latent_size, device, 
                                opt.is_bimanual_v2, opt.is_dgcnn, opt.is_bimanual_v3, use_test_reparam=opt.use_test_reparam,
                                is_bimanual=opt.is_bimanual, second_grasp_sample=opt.second_grasp_sample)
    elif arch == 'gan':
        net = GraspSamplerGAN(opt.model_scale, opt.pointnet_radius,
                              opt.pointnet_nclusters, opt.latent_size, device, opt.is_bimanual_v2, opt.is_bimanual_v3)
    elif arch == 'evaluator':
        if opt.is_bimanual:
            net = BimanualGraspEvaluator(opt.model_scale, opt.pointnet_radius, opt.pointnet_nclusters, device)
        else:
            net = GraspEvaluator(opt.model_scale, opt.pointnet_radius,
                                opt.pointnet_nclusters, device)
    else:
        raise NotImplementedError('model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    if opt.arch == 'vae':
        kl_loss = losses.kl_divergence
        reconstruction_loss = losses.control_point_l1_loss
        return kl_loss, reconstruction_loss
    elif opt.arch == 'gan':
        reconstruction_loss = losses.min_distance_loss
        return reconstruction_loss
    elif opt.arch == 'evaluator':
        loss = losses.classification_with_confidence_loss
        return loss
    else:
        raise NotImplementedError("Loss not found")


class GraspSampler(nn.Module):
    def __init__(self, latent_size, device):
        super(GraspSampler, self).__init__()
        self.latent_size = latent_size
        self.device = device

    def create_decoder(self, model_scale, pointnet_radius, pointnet_nclusters,
                       num_input_features, is_bimanual_v2=False, is_dgcnn=False, is_bimanual_v3=False,
                       is_bimanual=False):
        # The number of input features for the decoder is 3+latent space where 3
        # represents the x, y, z position of the point-cloud
        if not is_bimanual_v2:
            if is_dgcnn:
                self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                            model_scale, num_input_features, is_dgcnn=True)
                self.q = nn.Linear(model_scale * 1024, 4)
                self.t = nn.Linear(model_scale * 1024, 3)
                self.confidence = nn.Linear(model_scale * 1024, 1)
            elif is_bimanual:
                self.decoder =base_network(pointnet_radius, pointnet_nclusters,
                                            model_scale, num_input_features)
                self.dir1_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                self.app1_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                self.point1_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                self.confidence = nn.Linear(512, 1)
                
            else:
                self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                            model_scale, num_input_features)
                self.q = nn.Linear(model_scale * 1024, 4)
                self.t = nn.Linear(model_scale * 1024, 3)
                self.confidence = nn.Linear(model_scale * 1024, 1)
        else:
            if is_bimanual_v3:
                self.decoder =base_network(pointnet_radius, pointnet_nclusters,
                                            model_scale, num_input_features)
                # will skip linear layer in self.decoder, output would be 512
                self.dir1_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                
                self.dir2_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                
                self.app1_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                
                self.app2_layer = nn.Sequential(nn.Conv1d(512, 256, 1),
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                
                self.point1_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                
                self.point2_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                                nn.BatchNorm1d(256), nn.ReLU(True),
                                                nn.Conv1d(256, 128, 1),
                                                nn.BatchNorm1d(128), nn.ReLU(True),
                                                nn.Conv1d(128, 3, 1))
                
                self.confidence = nn.Linear(512, 1)
                
                
            else:
                self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                            model_scale, num_input_features)
                self.q1 = nn.Linear(model_scale * 1024, 4)
                self.t1 = nn.Linear(model_scale * 1024, 3)
                self.q2 = nn.Linear(model_scale * 1024, 4)
                self.t2 = nn.Linear(model_scale * 1024, 3)
                self.confidence = nn.Linear(model_scale * 1024, 1)
            
    def decode(self, xyz, z, is_bimanual_v2=False, is_dgcnn=False, is_bimanual_v3=False, is_bimanual=False, 
               second_grasp_sample=False, first_grasp=None, use_block=False, features=None):
        if not is_bimanual_v2:
            if is_dgcnn:
                xyz_features = z.unsqueeze(1).expand(-1, xyz.shape[1], -1)
                xyz_features = torch.transpose(xyz_features, 1, 2).contiguous()
                xyz_features = self.decoder[0](xyz, xyz_features)
                x = self.decoder[1](xyz_features.squeeze(-1))
            
            elif is_bimanual:
                if use_block:
                    pc = xyz
                    origin_features = features
                    predicted_dir1_list = []
                    predicted_app1_list = []
                    predicted_point1_list = []
                    predicted_confidence_list = []
                    for block_idx in range(z.shape[0]):
                        xyz = pc
                        xyz_features = origin_features
                        
                        xyz_features = self.concatenate_z_with_pc(xyz,z[block_idx])
                        if features is not None:
                            xyz_features = torch.cat((xyz_features, features[block_idx]), -1)
                        xyz_features = xyz_features.transpose(-1,1).contiguous()
                        for module in self.decoder[0]:
                            xyz, xyz_features = module(xyz, xyz_features)
                        predicted_dir1 = F.normalize(self.dir1_layer(xyz_features),p=2,dim=1).squeeze(-1)
                        predicted_app1 = F.normalize(self.app1_layer(xyz_features),p=2,dim=1).squeeze(-1)
                        predicted_point1 = self.point1_layer(xyz_features).squeeze(-1)
                        predicted_confidence = torch.sigmoid(self.confidence(xyz_features.squeeze(-1))).squeeze()
                        
                        predicted_dir1_list.append(predicted_dir1)
                        predicted_app1_list.append(predicted_app1)
                        predicted_point1_list.append(predicted_point1)
                        predicted_confidence_list.append(predicted_confidence)
                    
                    predicted_dir1_list = torch.stack(predicted_dir1_list, dim=0)
                    predicted_app1_list = torch.stack(predicted_app1_list, dim=0)
                    predicted_point1_list = torch.stack(predicted_point1_list, dim=0)
                    predicted_confidence_list = torch.stack(predicted_confidence_list, dim=0)
                    
                    return predicted_dir1_list, predicted_app1_list, predicted_point1_list, predicted_confidence_list
                
                else:
                    if second_grasp_sample:
                        xyz_features = self.concatenate_z_with_pc(xyz,z)
                        xyz_features = torch.cat((xyz_features, first_grasp.unsqueeze(1).expand(-1, xyz.shape[1], -1)), -1).transpose(-1,1).contiguous()
                    else:
                        xyz_features = self.concatenate_z_with_pc(xyz,z).transpose(-1,1).contiguous()

                    for module in self.decoder[0]:
                        xyz, xyz_features = module(xyz, xyz_features)
                    predicted_dir1 = F.normalize(self.dir1_layer(xyz_features),p=2,dim=1).squeeze(-1)
                    predicted_app1 = F.normalize(self.app1_layer(xyz_features),p=2,dim=1).squeeze(-1)
                    predicted_point1 = self.point1_layer(xyz_features).squeeze(-1)
                    predicted_confidence = torch.sigmoid(self.confidence(xyz_features.squeeze(-1))).squeeze()
                    return predicted_dir1, predicted_app1, predicted_point1, predicted_confidence
                
            else:
                xyz_features = self.concatenate_z_with_pc(xyz,z).transpose(-1,1).contiguous()
                for module in self.decoder[0]:
                    xyz, xyz_features = module(xyz, xyz_features)
                x = self.decoder[1](xyz_features.squeeze(-1))
            
            predicted_qt = torch.cat(
                (F.normalize(self.q(x), p=2, dim=-1), self.t(x)), -1)
            return predicted_qt, torch.sigmoid(self.confidence(x)).squeeze()
        else:
            if is_bimanual_v3:
                xyz_features = self.concatenate_z_with_pc(xyz,z).transpose(-1,1).contiguous()
                for module in self.decoder[0]:
                    xyz, xyz_features = module(xyz, xyz_features)

                predicted_dir1 = F.normalize(self.dir1_layer(xyz_features),p=2,dim=1).squeeze(-1)
                predicted_dir2 = F.normalize(self.dir2_layer(xyz_features),p=2,dim=1).squeeze(-1)
                predicted_app1 = F.normalize(self.app1_layer(xyz_features),p=2,dim=1).squeeze(-1)
                predicted_app2 = F.normalize(self.app2_layer(xyz_features),p=2,dim=1).squeeze(-1)
                predicted_point1 = self.point1_layer(xyz_features).squeeze(-1)
                predicted_point2 = self.point2_layer(xyz_features).squeeze(-1)

                predicted_confidence = torch.sigmoid(self.confidence(xyz_features.squeeze(-1))).squeeze()

                return predicted_dir1, predicted_dir2, predicted_app1, predicted_app2, predicted_point1, predicted_point2, predicted_confidence
            else:
                xyz_features = self.concatenate_z_with_pc(xyz,z).transpose(-1,1).contiguous()
                for module in self.decoder[0]:
                    xyz, xyz_features = module(xyz, xyz_features)

                x = self.decoder[1](xyz_features.squeeze(-1))
                predicted_qt1 = torch.cat((F.normalize(self.q1(x), p=2, dim=-1), self.t1(x)), -1)
                predicted_qt2 = torch.cat((F.normalize(self.q2(x), p=2, dim=-1), self.t2(x)), -1)
                predicted_qt = torch.cat((predicted_qt1.unsqueeze(0), predicted_qt2.unsqueeze(0)), dim=0)
                predicted_qt = torch.transpose(predicted_qt, 0, 1)

                return predicted_qt, torch.sigmoid(self.confidence(x)).squeeze()

        
    def concatenate_z_with_pc(self, pc, z):
        # z.unsqueeze_(1)
        z = z.unsqueeze(1)
        z = z.expand(-1, pc.shape[1], -1)
        return torch.cat((pc, z), -1)

    def get_latent_size(self):
        return self.latent_size


class GraspSamplerVAE(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 latent_size=2,
                 device="cpu",
                 is_bimanual_v2=False,
                 is_dgcnn=False,
                 is_bimanual_v3=False,
                 use_test_reparam=False,
                 is_bimanual=False,
                 second_grasp_sample=False):
        super(GraspSamplerVAE, self).__init__(latent_size, device)
        self.device = device

        self.is_bimanual_v2 = is_bimanual_v2
        self.is_bimanual_v3 = is_bimanual_v3
        self.is_dgcnn = is_dgcnn
        self.use_test_reparam = use_test_reparam
        self.is_bimanual = is_bimanual
        self.second_grasp_sample = second_grasp_sample
        
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters, self.is_dgcnn)

        if self.is_dgcnn:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size, self.is_bimanual_v2, is_dgcnn=True)
        elif self.is_bimanual_v2 and not self.is_bimanual_v3:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size + 3, self.is_bimanual_v2)
        elif self.is_bimanual_v3:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size+3, self.is_bimanual_v2, is_bimanual_v3=True)
        elif self.second_grasp_sample:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,latent_size+19, is_bimanual=self.is_bimanual)
        else:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters, latent_size+3, is_bimanual=self.is_bimanual)
        
        self.create_bottleneck(model_scale * 1024, latent_size)
                
    def create_encoder(
            self,
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
            bimanual=False):
        # The number of input features for the encoder is 19: the x, y, z
        # position of the point-cloud and the flattened 4x4=16 grasp pose matrix
        if self.is_bimanual_v2:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 35)
        elif self.is_dgcnn:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 16, is_dgcnn=True, device=self.device)
        elif self.second_grasp_sample:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 35)
        else:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 19)

    def create_bottleneck(self, input_size, latent_size):
        mu = nn.Linear(input_size, latent_size)
        logvar = nn.Linear(input_size, latent_size)
        self.latent_space = nn.ModuleList([mu, logvar])

    def encode(self, xyz, xyz_features):
        if self.is_dgcnn:
            xyz_features = self.encoder[0](xyz, xyz_features)
            return self.encoder[1](xyz_features.squeeze(-1))
        else:
            for module in self.encoder[0]:
                # print()
                # print(xyz.type(), xyz_features.type())
                # print('xyz: ', xyz.shape, 'xyz_features: ', xyz_features.shape)
                xyz, xyz_features = module(xyz, xyz_features)
            # print(xyz_features.shape)
            # exit() 
            return self.encoder[1](xyz_features.squeeze(-1))

    def bottleneck(self, z):
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, pc, grasp=None, train=True):
        if train:
            return self.forward_train(pc, grasp)
        else:
            return self.forward_test(pc, grasp)

    def forward_train(self, pc, grasp):
        if self.is_dgcnn:
            input_features = grasp.unsqueeze(1).expand(-1, pc.shape[1], -1) # (64, 1024, 16)
            input_features = torch.transpose(input_features, 1, 2).contiguous() # (64, 16, 1024)
        else:
            input_features = torch.cat(
                (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
        start = time()
        z = self.encode(pc, input_features) #(64, 1024)
        end = time()
        # print('encode time', end - start)
        mu, logvar = self.bottleneck(z)

        z = self.reparameterize(mu, logvar)
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence, mu, logvar
        elif self.is_bimanual and not self.second_grasp_sample:
            dir1, app1, point1, confidence = self.decode(pc, z, is_bimanual=self.is_bimanual)
            return dir1, app1, point1, confidence, mu, logvar
        elif self.second_grasp_sample:
            first_grasp = grasp[:, :16]
            dir2, app2, point2, confidence = self.decode(pc, z, is_bimanual=self.is_bimanual, second_grasp_sample=self.second_grasp_sample, first_grasp=first_grasp)
            return dir2, app2, point2, confidence, mu, logvar
        else:
            qt, confidence = self.decode(pc, z, self.is_bimanual_v2, self.is_dgcnn)
            return qt, confidence, mu, logvar
            
    def forward_test(self, pc, grasp):
        if self.is_dgcnn:
            input_features = grasp.unsqueeze(1).expand(-1, pc.shape[1], -1) # (64, 1024, 16)
            input_features = torch.transpose(input_features, 1, 2).contiguous() # (64, 16, 1024)
        else:
            input_features = torch.cat(
                (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
        
        z = self.encode(pc, input_features)
        # print(z.shape)
        mu, logvar = self.bottleneck(z)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>check device',z.device, pc.device)
        #* check if we use logvar when testing
        if self.use_test_reparam:
            mu = self.reparameterize(mu, logvar) # (96, 5)
        
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, mu, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence
        elif self.is_bimanual and not self.second_grasp_sample:
            dir1, app1, point1, confidence = self.decode(pc, mu, is_bimanual=self.is_bimanual)
            return dir1, app1, point1, confidence
        elif self.second_grasp_sample:
            first_grasp = grasp[:, :16]
            dir2, app2, point2, confidence = self.decode(pc, mu, is_bimanual=self.is_bimanual, second_grasp_sample=self.second_grasp_sample, first_grasp=first_grasp)
            return dir2, app2, point2, confidence
        else:
            qt, confidence = self.decode(pc, mu, self.is_bimanual_v2, self.is_dgcnn)
            return qt, confidence

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None):
        if z is None:
            z = self.sample_latent(pc.shape[0])
        if self.is_bimanual:
            dir1, app1, point1, confidence = self.decode(pc, z, is_bimanual=self.is_bimanual)
            return dir1, app1, point1, confidence, z.squeeze()
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence, z.squeeze()
        else:
            qt, confidence = self.decode(pc, z, self.is_bimanual_v2)
            return qt, confidence, z.squeeze()

    def generate_dense_latents(self, resolution):
        """
        For the VAE sampler we consider dense latents to correspond to those between -2 and 2
        """
        latents = torch.meshgrid(*[
            torch.linspace(-2, 2, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1).to(self.device)

class GraspSamplerCrossConditionVAE(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 latent_size=2,
                 device="cpu",
                 cross_condition=False):
        super(GraspSamplerCrossConditionVAE, self).__init__(latent_size, device)
        self.device = device

        # self.is_bimanual_v2 = is_bimanual_v2
        # self.is_bimanual_v3 = is_bimanual_v3
        # self.is_dgcnn = is_dgcnn
        # self.use_test_reparam = use_test_reparam
        # self.is_bimanual = is_bimanual
        # self.second_grasp_sample = second_grasp_sample
        
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters)
        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters, num_input_features=latent_size+3)
        # if self.is_dgcnn:
        #     self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
        #                         latent_size, self.is_bimanual_v2, is_dgcnn=True)
        # elif self.is_bimanual_v2 and not self.is_bimanual_v3:
        #     self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
        #                         latent_size + 3, self.is_bimanual_v2)
        # elif self.is_bimanual_v3:
        #     self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
        #                         latent_size+3, self.is_bimanual_v2, is_bimanual_v3=True)
        # elif self.second_grasp_sample:
        #     self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,latent_size+19, is_bimanual=self.is_bimanual)
        # else:
        #     self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters, latent_size+3, is_bimanual=self.is_bimanual)
        
        self.create_bottleneck(model_scale * 1024, latent_size)
                
    def create_encoder(
            self,
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
            bimanual=False):
        # The number of input features for the encoder is 19: the x, y, z
        # position of the point-cloud and the flattened 4x4=16 grasp pose matrix
        # if self.is_bimanual_v2:
        #     self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 35)
        # elif self.is_dgcnn:
        #     self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 16, is_dgcnn=True, device=self.device)
        # elif self.second_grasp_sample:
        #     self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 35)
        # else:
        #     self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 19)
        self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 35)

    def create_decoder(self, model_scale, pointnet_radius, pointnet_nclusters, num_input_features):
        #* create decoders for grasp pair generator and single grasp genertor conditioned with grasp pair
        # for grasp pair generator
        self.decoder1 = base_network(pointnet_radius, pointnet_nclusters, model_scale, num_input_features)
        self.dir11_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        
        self.dir12_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        
        self.app11_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        
        self.app12_layer = nn.Sequential(nn.Conv1d(512, 256, 1),
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        
        self.point11_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        
        self.point12_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        self.confidence1 = nn.Linear(512, 1)
        #for single grasp generator conditioned with grasp pair
        self.decoder2 = base_network(pointnet_radius, pointnet_nclusters, model_scale, num_input_features+16)
        self.dir2_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        self.app2_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        self.point2_layer = nn.Sequential(nn.Conv1d(512, 256, 1), 
                                        nn.BatchNorm1d(256), nn.ReLU(True),
                                        nn.Conv1d(256, 128, 1),
                                        nn.BatchNorm1d(128), nn.ReLU(True),
                                        nn.Conv1d(128, 3, 1))
        self.confidence2 = nn.Linear(512, 1)




    def create_bottleneck(self, input_size, latent_size):
        mu1 = nn.Linear(input_size, latent_size)
        logvar1 = nn.Linear(input_size, latent_size)
        self.latent_space1 = nn.ModuleList([mu1, logvar1])
        
        mu2 = nn.Linear(input_size, latent_size)
        logvar2 = nn.Linear(input_size, latent_size)
        self.latent_space2 = nn.ModuleList([mu2, logvar2])

    def encode(self, xyz, xyz_features):
        
        for module in self.encoder[0]:
            # print()
            # print(xyz.type(), xyz_features.type())
            xyz, xyz_features = module(xyz, xyz_features)
        return self.encoder[1](xyz_features.squeeze(-1))

    def decode1(self, xyz, z):
        xyz_features = self.concatenate_z_with_pc(xyz, z).transpose(-1, 1).contiguous()
        for module in self.decoder1[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        predicted_dir11 = F.normalize(self.dir11_layer(xyz_features),p=2,dim=1).squeeze(-1)
        predicted_dir12 = F.normalize(self.dir12_layer(xyz_features),p=2,dim=1).squeeze(-1)
        predicted_app11 = F.normalize(self.app11_layer(xyz_features),p=2,dim=1).squeeze(-1)
        predicted_app12 = F.normalize(self.app12_layer(xyz_features),p=2,dim=1).squeeze(-1)
        predicted_point11 = self.point11_layer(xyz_features).squeeze(-1)
        predicted_point12 = self.point12_layer(xyz_features).squeeze(-1)
        
        predicted_confidence = torch.sigmoid(self.confidence1(xyz_features.squeeze(-1))).squeeze()
        return predicted_dir11, predicted_dir12, predicted_app11, predicted_app12,\
                predicted_point11, predicted_point12, predicted_confidence
        
    def decode2(self, xyz, z, condition_grasp):
        xyz_features = self.concatenate_z_with_pc(xyz, z)
        xyz_features = torch.cat((xyz_features, condition_grasp.unsqueeze(1).expand(-1, xyz.shape[1], -1)), -1).transpose(-1,1).contiguous()
        for module in self.decoder2[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        predicted_dir2 = F.normalize(self.dir2_layer(xyz_features),p=2,dim=1).squeeze(-1)
        predicted_app2 = F.normalize(self.app2_layer(xyz_features),p=2,dim=1).squeeze(-1)
        predicted_point2 = F.normalize(self.point2_layer(xyz_features),p=2,dim=1).squeeze(-1)
        
        predicted_confidence2 = torch.sigmoid(self.confidence2(xyz_features.squeeze(-1))).squeeze()
        return predicted_dir2, predicted_app2, predicted_point2, predicted_confidence2
        
    
    def bottleneck1(self, z):
        return self.latent_space1[0](z), self.latent_space1[1](z)

    def bottleneck2(self, z):
        return self.latent_space2[0](z), self.latent_space2[1](z)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, pc, grasp=None, train=True):
        if train:
            return self.forward_train(pc, grasp)
        else:
            return self.forward_test(pc, grasp)

    def forward_train(self, pc, grasp):
        
        # grasp pair generate
        input_features = torch.cat((pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)), -1).transpose(-1, 1).contiguous()
        z = self.encode(pc, input_features) #(B, 1024)
        mu, logvar = self.bottleneck1(z)
        z = self.reparameterize(mu, logvar)
        dir11, dir12, app11, app12, point11, point12, confidence1 = self.decode1(pc, z)
        # single grasp generate conditioned with grasp pair, generate grasp 1 first then generate grasp 2
        grasps1 = utils.get_homog_matrix(app11, dir11, point11, app11.shape[0], device=self.device)
        grasps2 = utils.get_homog_matrix(app12, dir12, point12, app12.shape[0], device=self.device)
        grasps1 = grasps1.reshape(-1, 16)
        grasps2 = grasps2.reshape(-1, 16)
        pred_grasps = torch.cat([grasps1, grasps2], dim=1)
        # pred_grasps = pred_grasps.requires_grad_(True)

        input_features2 = torch.cat((pc, pred_grasps.unsqueeze(1).expand(-1, pc.shape[1], -1)), -1).transpose(-1, 1).contiguous()
        z1 = self.encode(pc, input_features2) #(B, 1024)
        mu1, logvar1 = self.bottleneck2(z1)
        z1 = self.reparameterize(mu1, logvar1)
        # generate grasp1
        dir21, app21, point21, confidence21 = self.decode2(pc, z1, grasps2)
        # generate grasp2
        dir22, app22, point22, confidence22 = self.decode2(pc, z1, grasps1)
        
        return [dir11, dir12, app11, app12, point11, point12, confidence1, mu, logvar],\
                [dir21, dir22, app21, app22, point21, point22, confidence21, confidence22, mu1, logvar1]
        # if self.is_dgcnn:
        #     input_features = grasp.unsqueeze(1).expand(-1, pc.shape[1], -1) # (64, 1024, 16)
        #     input_features = torch.transpose(input_features, 1, 2).contiguous() # (64, 16, 1024)
        # else:
        #     input_features = torch.cat(
        #         (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
        #         -1).transpose(-1, 1).contiguous()
        # start = time()
        # z = self.encode(pc, input_features) #(64, 1024)
        # end = time()
        # # print('encode time', end - start)
        # mu, logvar = self.bottleneck(z)

        # z = self.reparameterize(mu, logvar)
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence, mu, logvar
        elif self.is_bimanual and not self.second_grasp_sample:
            dir1, app1, point1, confidence = self.decode(pc, z, is_bimanual=self.is_bimanual)
            return dir1, app1, point1, confidence, mu, logvar
        elif self.second_grasp_sample:
            first_grasp = grasp[:, :16]
            dir2, app2, point2, confidence = self.decode(pc, z, is_bimanual=self.is_bimanual, second_grasp_sample=self.second_grasp_sample, first_grasp=first_grasp)
            return dir2, app2, point2, confidence, mu, logvar
        else:
            qt, confidence = self.decode(pc, z, self.is_bimanual_v2, self.is_dgcnn)
            return qt, confidence, mu, logvar
            
    def forward_test(self, pc, grasp):
        
        input_features = torch.cat((pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)), -1).transpose(-1, 1).contiguous()
        z = self.encode(pc, input_features) #(B, 1024)
        mu, logvar = self.bottleneck1(z)
        
        dir11, dir12, app11, app12, point11, point12, confidence1 = self.decode1(pc, mu)
        
        grasps1 = utils.get_homog_matrix(app11, dir11, point11, app11.shape[0], device=self.device)
        grasps2 = utils.get_homog_matrix(app12, dir12, point12, app12.shape[0], device=self.device)
        grasps1 = grasps1.reshape(-1, 16)
        grasps2 = grasps2.reshape(-1, 16)
        pred_grasps = torch.cat([grasps1, grasps2], dim=1)
        
        input_features2 = torch.cat((pc, pred_grasps.unsqueeze(1).expand(-1, pc.shape[1], -1)), -1).transpose(-1, 1).contiguous()
        z1 = self.encode(pc, input_features2) #(B, 1024)
        mu1, logvar1 = self.bottleneck2(z1)
        
        # generate grasp1
        dir21, app21, point21, confidence21 = self.decode2(pc, mu1, grasps2)
        # generate grasp2
        dir22, app22, point22, confidence22 = self.decode2(pc, mu1, grasps1)
        
        return [dir11, dir12, app11, app12, point11, point12, confidence1],\
                [dir21, dir22, app21, app22, point21, point22, confidence21, confidence22]
        
        # if self.is_dgcnn:
        #     input_features = grasp.unsqueeze(1).expand(-1, pc.shape[1], -1) # (64, 1024, 16)
        #     input_features = torch.transpose(input_features, 1, 2).contiguous() # (64, 16, 1024)
        # else:
        #     input_features = torch.cat(
        #         (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
        #         -1).transpose(-1, 1).contiguous()
        
        z = self.encode(pc, input_features)
        # print(z.shape)
        mu, logvar = self.bottleneck(z)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>check device',z.device, pc.device)
        #* check if we use logvar when testing
        if self.use_test_reparam:
            mu = self.reparameterize(mu, logvar) # (96, 5)
        
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, mu, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence
        elif self.is_bimanual and not self.second_grasp_sample:
            dir1, app1, point1, confidence = self.decode(pc, mu, is_bimanual=self.is_bimanual)
            return dir1, app1, point1, confidence
        elif self.second_grasp_sample:
            first_grasp = grasp[:, :16]
            dir2, app2, point2, confidence = self.decode(pc, mu, is_bimanual=self.is_bimanual, second_grasp_sample=self.second_grasp_sample, first_grasp=first_grasp)
            return dir2, app2, point2, confidence
        else:
            qt, confidence = self.decode(pc, mu, self.is_bimanual_v2, self.is_dgcnn)
            return qt, confidence

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None):
        if z is None:
            z = self.sample_latent(pc.shape[0])
        if self.is_bimanual:
            dir1, app1, point1, confidence = self.decode(pc, z, is_bimanual=self.is_bimanual)
            return dir1, app1, point1, confidence, z.squeeze()
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence, z.squeeze()
        else:
            qt, confidence = self.decode(pc, z, self.is_bimanual_v2)
            return qt, confidence, z.squeeze()

    def generate_dense_latents(self, resolution):
        """
        For the VAE sampler we consider dense latents to correspond to those between -2 and 2
        """
        latents = torch.meshgrid(*[
            torch.linspace(-2, 2, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1).to(self.device)


class GraspSamplerGAN(GraspSampler):
    """
    Altough the name says this sampler is based on the GAN formulation, it is
    not actually optimizing based on the commonly known adversarial game.
    Instead, it is based on the Implicit Maximum Likelihood Estimation from
    https://arxiv.org/pdf/1809.09087.pdf which is similar to the GAN formulation
    but with new insights that avoids e.g. mode collapses.
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius,
                 pointnet_nclusters,
                 latent_size=2,
                 device="cpu",
                 is_bimanual_v2=False,
                 is_bimanual_v3=False):
        super(GraspSamplerGAN, self).__init__(latent_size, device)
        self.is_bimanual_v2 = is_bimanual_v2
        self.is_bimanual_v3 = is_bimanual_v3

        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                            latent_size + 3)
        if self.is_bimanual_v2 and not self.is_bimanual_v3:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size + 3, self.is_bimanual_v2)
        elif self.is_bimanual_v3:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size+3, self.is_bimanual_v2, is_bimanual_v3=True)

    def sample_latent(self, batch_size, device=None):
        return torch.rand(batch_size, self.latent_size).to(device)

    def forward(self, pc, grasps=None, train=True):
        z = self.sample_latent(pc.shape[0], device=pc.device)
        return self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=self.is_bimanual_v3)

    def generate_grasps(self, pc, z=None):
        if z is None:
            z = self.sample_latent(pc.shape[0])
        qt, confidence = self.decode(pc, z)
        return qt, confidence, z.squeeze()

    def generate_dense_latents(self, resolution):
        latents = torch.meshgrid(*[
            torch.linspace(0, 1, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1).to(self.device)


class GraspEvaluator(nn.Module):
    def __init__(self,
                 model_scale=1,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 device="cpu"):
        super(GraspEvaluator, self).__init__()
        self.create_evaluator(pointnet_radius, model_scale, pointnet_nclusters)
        self.device = device

    def create_evaluator(self, pointnet_radius, model_scale,
                         pointnet_nclusters):
        # The number of input features for the evaluator is 4: the x, y, z
        # position of the concatenated gripper and object point-clouds and an
        # extra binary feature, which is 0 for the object and 1 for the gripper,
        # to tell these point-clouds apart
        self.evaluator = base_network(pointnet_radius, pointnet_nclusters,
                                      model_scale, 4)
        self.predictions_logits = nn.Linear(1024 * model_scale, 1)
        self.confidence = nn.Linear(1024 * model_scale, 1)

    def evaluate(self, xyz, xyz_features):
        for module in self.evaluator[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        return self.evaluator[1](xyz_features.squeeze(-1))

    def forward(self, pc, gripper_pc, train=True):

        pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
        x = self.evaluate(pc, pc_features.contiguous())
        return self.predictions_logits(x), torch.sigmoid(self.confidence(x))

    def merge_pc_and_gripper_pc(self, pc, gripper_pc):
        """
        Merges the object point cloud and gripper point cloud and
        adds a binary auxiliary feature that indicates whether each point
        belongs to the object or to the gripper.
        """
        pc_shape = pc.shape
        gripper_shape = gripper_pc.shape
        assert (len(pc_shape) == 3)
        assert (len(gripper_shape) == 3)
        assert (pc_shape[0] == gripper_shape[0])

        npoints = pc_shape[1]
        batch_size = pc_shape[0]

        l0_xyz = torch.cat((pc, gripper_pc), 1)
        labels = [
            torch.ones(pc.shape[1], 1, dtype=torch.float32),
            torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)
        ]

        labels = torch.cat(labels, 0)
        labels.unsqueeze_(0)
        labels = labels.repeat(batch_size, 1, 1)

        l0_points = torch.cat([l0_xyz, labels.to(self.device)],
                              -1).transpose(-1, 1)

        
        return l0_xyz, l0_points
    
    
class BimanualGraspEvaluator(nn.Module):
    def __init__(self,
                 model_scale=1,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 device="cpu"):
        super(BimanualGraspEvaluator, self).__init__()
        self.create_evaluator(pointnet_radius, model_scale, pointnet_nclusters)
        self.device = device

    def create_evaluator(self, pointnet_radius, model_scale,
                         pointnet_nclusters):
        # The number of input features for the evaluator is 4: the x, y, z
        # position of the concatenated gripper and object point-clouds and an
        # extra binary feature, which is 0 for the object and 1 for the gripper,
        # to tell these point-clouds apart
        self.evaluator = base_network(pointnet_radius, pointnet_nclusters,
                                      model_scale, 4)
        self.predictions_logits = nn.Linear(1024 * model_scale, 1)
        self.confidence = nn.Linear(1024 * model_scale, 1)

    def evaluate(self, xyz, xyz_features):
        for module in self.evaluator[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        return self.evaluator[1](xyz_features.squeeze(-1))

    def forward(self, pc, gripper_pc, train=True):

        pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
        x = self.evaluate(pc, pc_features.contiguous())
        return self.predictions_logits(x), torch.sigmoid(self.confidence(x))

    def merge_pc_and_gripper_pc(self, pc, gripper_pc):
        """
        Merges the object point cloud and gripper point cloud and
        adds a binary auxiliary feature that indicates whether each point
        belongs to the object or to the gripper.
        """
        pc_shape = pc.shape
        gripper_shape = gripper_pc.shape
        assert (len(pc_shape) == 3)
        assert (len(gripper_shape) == 3)
        assert (pc_shape[0] == gripper_shape[0])

        npoints = pc_shape[1]
        batch_size = pc_shape[0]

        l0_xyz = torch.cat((pc, gripper_pc), 1)
        labels = [
            torch.ones(pc.shape[1], 1, dtype=torch.float32),
            torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)
        ]
        labels = torch.cat(labels, 0)
        labels.unsqueeze_(0)
        labels = labels.repeat(batch_size, 1, 1)

        l0_points = torch.cat([l0_xyz, labels.to(l0_xyz.device)],
                              -1).transpose(-1, 1)
        
        return l0_xyz, l0_points    
    

class GraspSamplerVAEBlock(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 latent_size=2,
                 device="cpu",
                 is_bimanual_v2=False,
                 is_bimanual_v3=False):
        super(GraspSamplerVAEBlock, self).__init__(latent_size, device)
        self.device = device

        self.is_bimanual_v2 = is_bimanual_v2
        self.is_bimanual_v3 = is_bimanual_v3
        self.npoints = 2048
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters)

        
        if self.is_bimanual_v2 and not self.is_bimanual_v3:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size + 3, self.is_bimanual_v2)
        elif self.is_bimanual_v3:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size+3, self.is_bimanual_v2, is_bimanual_v3=True)
        else:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters, latent_size+3+1, is_bimanual=True)
        
        self.create_bottleneck(model_scale * 1024, latent_size)
                
    def create_encoder(
            self,
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
            bimanual=False):
        # The number of input features for the encoder is 19: the x, y, z
        # position of the point-cloud and the flattened 4x4=16 grasp pose matrix
        if self.is_bimanual_v2:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters, 
                                        model_scale, 35)
        # elif self.is_dgcnn:
        #     self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 16, is_dgcnn=True, device=self.device)
        else:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters,
                                        model_scale, 20)
        
        # self.encoder = base_network(pointnet_radius, pointnet_nclusters,
        #                             model_scale, 16)

    def create_bottleneck(self, input_size, latent_size):
        mu = nn.Linear(input_size, latent_size)
        logvar = nn.Linear(input_size, latent_size)
        self.latent_space = nn.ModuleList([mu, logvar])

    def encode(self, xyz, grasp, xyz_features):
        
        z_list = []
        pc = xyz 
        features = xyz_features
        for block_idx in range(grasp.shape[0]):
            xyz = pc
            xyz_features = features

            input_features = torch.cat((xyz, grasp[block_idx].unsqueeze(1).expand(-1, xyz.shape[1], -1)),-1)
            xyz_features = torch.cat((input_features, xyz_features[block_idx]), -1)
            xyz_features = xyz_features.transpose(-1, 1).contiguous()

            for module in self.encoder[0]:
                # print()
                # print(xyz.type(), xyz_features.type())
                xyz, xyz_features = module(xyz, xyz_features)

            z = self.encoder[1](xyz_features.squeeze(-1))
            z_list.append(z)

        z_list = torch.stack(z_list, dim=0)

        return z_list

    def bottleneck(self, z):
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, z_list, num_block):
        mu_list = []
        logvar_list = []
        reparam_z_list = []
        for block_idx in range(num_block):
            mu, logvar = self.bottleneck(z_list[block_idx])
            mu_list.append(mu)
            logvar_list.append(logvar)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            reparam_z_list.append(mu + eps * std)
        
        mu_list = torch.stack(mu_list, dim=0)
        logvar_list = torch.stack(logvar_list, dim=0)
        reparam_z_list = torch.stack(reparam_z_list, dim=0)
        return reparam_z_list, mu_list, logvar_list
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # return mu + eps * std

    def forward(self, pc, grasp=None, train=True, features=None, targets=None):
        if train:
            return self.forward_train(pc, grasp, features=features, targets=targets)
        else:
            return self.forward_test(pc, grasp, features=features, targets=targets)

    def forward_train(self, pc, grasp, features=None, targets=None):

        grasp = grasp.transpose(0,1)
        features = features.transpose(0,1)
        targets = targets.transpose(0,1)
        z_list = self.encode(pc, grasp, features) #(64, 1024)

        # mu, logvar = self.bottleneck(z)
        z_list, mu_list, logvar_list = self.reparameterize(z_list, grasp.shape[0]) # (96, 5)
        
        dir1_list, app1_list, point1_list, confidence_list = self.decode(pc, z_list, is_bimanual=True, use_block=True, features=features)
        
        return dir1_list.transpose(0,1), app1_list.transpose(0,1), point1_list.transpose(0,1), \
               confidence_list.transpose(0,1), mu_list.transpose(0,1), logvar_list.transpose(0,1), targets.transpose(0,1)
        # if self.is_bimanual_v3:
        #     dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=True)
        #     return dir1, dir2, app1, app2, point1, point2, confidence, mu, logvar
        # else:
        #     qt, confidence = self.decode(pc, z, self.is_bimanual_v2, self.is_dgcnn)
        #     return qt, confidence, mu, logvar
            
    def forward_test(self, pc, grasp, features=None, targets=None):
        grasp = grasp.transpose(0,1)
        features = features.transpose(0,1)
        targets = targets.transpose(0,1)

        z_list = self.encode(pc, grasp, features) #(64, 1024)

        # mu, logvar = self.bottleneck(z)
        z_list, mu_list, logvar_list = self.reparameterize(z_list, grasp.shape[0]) # (96, 5)
        
        dir1_list, app1_list, point1_list, confidence_list = self.decode(pc, mu_list, is_bimanual=True, use_block=True, features=features)
        
        return dir1_list.transpose(0,1), app1_list.transpose(0,1), \
               point1_list.transpose(0,1), confidence_list.transpose(0,1), targets.transpose(0,1)
        
        if self.is_dgcnn:
            input_features = grasp.unsqueeze(1).expand(-1, pc.shape[1], -1) # (64, 1024, 16)
            input_features = torch.transpose(input_features, 1, 2).contiguous() # (64, 16, 1024)
        else:
            input_features = torch.cat(
                (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
                -1).transpose(-1, 1).contiguous()
        
        z = self.encode(pc, input_features)
        # print(z.shape)
        mu, _ = self.bottleneck(z)
        # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>check device',z.device, pc.device)
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, mu, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence
        else:
            qt, confidence = self.decode(pc, mu, self.is_bimanual_v2, self.is_dgcnn)
            return qt, confidence

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None):
        if z is None:
            z = self.sample_latent(pc.shape[0])
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence, z.squeeze()
        else:
            qt, confidence = self.decode(pc, z, self.is_bimanual_v2)
            return qt, confidence, z.squeeze()

    def generate_dense_latents(self, resolution):
        """
        For the VAE sampler we consider dense latents to correspond to those between -2 and 2
        """
        latents = torch.meshgrid(*[
            torch.linspace(-2, 2, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1).to(self.device)


class GraspSamplerVAEAnchor(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """
    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 latent_size=2,
                 device="cpu",
                 is_bimanual_v2=False,
                 is_bimanual_v3=False):
        super(GraspSamplerVAEAnchor, self).__init__(latent_size, device)
        self.device = device

        self.is_bimanual_v2 = is_bimanual_v2
        self.is_bimanual_v3 = is_bimanual_v3
        self.create_encoder(model_scale)
        self.backbone = base_network(pointnet_radius, pointnet_nclusters, model_scale, 0, use_anchor=True)
        self.create_anchor_layer(model_scale, input_size=3)
        self.create_decoder(model_scale)
        
        # if self.is_bimanual_v2 and not self.is_bimanual_v3:
        #     self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
        #                         latent_size + 3, self.is_bimanual_v2)
        # elif self.is_bimanual_v3:
        #     self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
        #                         latent_size+3, self.is_bimanual_v2, is_bimanual_v3=True)
        
        
        self.create_bottleneck(32, 32)
    
    def create_anchor_layer(self, model_scale, input_size=3):
        anchor_layer = nn.Sequential(nn.Linear(input_size*model_scale, 16*model_scale),
                                     nn.ReLU(),
                                     nn.Linear(16*model_scale, 32*model_scale),
                                     nn.ReLU(),
                                     nn.Linear(32*model_scale, 64*model_scale))
        self.anchor_layer = nn.ModuleList([anchor_layer])
         
    def create_encoder(self,model_scale):
        # The number of input features for the encoder is 19: the x, y, z
        # position of the point-cloud and the flattened 4x4=16 grasp pose matrix
        self.mlp1 = nn.Linear(1088, 128)
        self.mlp2 = nn.Linear(128, 32)
        self.mlp3 = nn.Linear(32, 32)
        
        encoder = nn.Sequential(self.mlp1, 
                                nn.BatchNorm1d(128), nn.ReLU(),
                                self.mlp2, 
                                nn.BatchNorm1d(32), nn.ReLU(),
                                self.mlp3)
        self.encoder = nn.ModuleList([encoder])
    
    def create_decoder(self, model_scale):
        hidden_dim = 128
        decoder = nn.Sequential(nn.Linear(1056, hidden_dim),
                                nn.Linear(hidden_dim, 3))
        self.confidence = nn.Linear(1056, 1)
        
        self.decoder = nn.ModuleList([decoder])
    
    def create_bottleneck(self, input_size, latent_size):
        mu = nn.Linear(input_size, latent_size)
        logvar = nn.Linear(input_size, latent_size)
        self.latent_space = nn.ModuleList([mu, logvar])

    def encode(self, xyz, xyz_features, grasp):
        # if self.is_dgcnn:
        #     xyz_features = self.backbone[0](xyz, xyz_features)
        #     return self.backbone[1](xyz_features.squeeze(-1))
        # else:
        for module in self.backbone[0]:
            # print()
            # print(xyz.type(), xyz_features.type())
            xyz, xyz_features = module(xyz, xyz_features)
        pc_feat = self.backbone[1](xyz_features.squeeze(-1))
        return pc_feat

            
            
    def bottleneck(self, z):
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pc, grasp=None, train=True):
        if train:
            return self.forward_train(pc, grasp)
        else:
            return self.forward_test(pc, grasp)

    def forward_train(self, pc, grasp):
        # if self.is_dgcnn:
        #     input_features = grasp.unsqueeze(1).expand(-1, pc.shape[1], -1) # (64, 1024, 16)
        #     input_features = torch.transpose(input_features, 1, 2).contiguous() # (64, 16, 1024)
        # else:
        #     input_features = torch.cat(
        #         (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
        #         -1).transpose(-1, 1).contiguous()

        # z = self.encode(pc, input_features) #(64, 1024)
        pc_feat = self.encode(pc, None, grasp)
        anchor_feat = self.anchor_layer[0](grasp)
        x = torch.cat((pc_feat, anchor_feat), dim=1)
        x = self.encoder[0](x)
    
        mu, logvar = self.bottleneck(x)
        z = self.reparameterize(mu, logvar) # (96, 5)
        z = torch.cat((z, pc_feat), dim=1)
        out = self.decoder[0](z)
        confidence = torch.sigmoid(self.confidence(z)).squeeze()
        
        return out, confidence, mu, logvar
    
        # if self.is_bimanual_v3:
        #     dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=True)
        #     return dir1, dir2, app1, app2, point1, point2, confidence, mu, logvar
        # else:
        #     qt, confidence = self.decode(pc, z, self.is_bimanual_v2, self.is_dgcnn)
        #     return qt, confidence, mu, logvar
            
    def forward_test(self, pc, grasp):
        # if self.is_dgcnn:
        #     input_features = grasp.unsqueeze(1).expand(-1, pc.shape[1], -1) # (64, 1024, 16)
        #     input_features = torch.transpose(input_features, 1, 2).contiguous() # (64, 16, 1024)
        # else:
        # input_features = torch.cat(
        #     (pc, grasp.unsqueeze(1).expand(-1, pc.shape[1], -1)),
        #     -1).transpose(-1, 1).contiguous()
        
        pc_feat = self.encode(pc, None, grasp)
        anchor_feat = self.anchor_layer[0](grasp)
        x = torch.cat((pc_feat, anchor_feat), dim=1)
        x = self.encoder[0](x)
        
        # print(z.shape)
        mu, _ = self.bottleneck(x)
        mu = torch.cat((mu, pc_feat), dim=1)
        out = self.decoder[0](mu)
        confidence = torch.sigmoid(self.confidence(mu)).squeeze()
        
        return out, confidence
    
        # # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>check device',z.device, pc.device)
        # if self.is_bimanual_v3:
        #     dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, mu, self.is_bimanual_v2, is_bimanual_v3=True)
        #     return dir1, dir2, app1, app2, point1, point2, confidence
        # else:
        #     qt, confidence = self.decode(pc, mu, self.is_bimanual_v2, self.is_dgcnn)
        #     return qt, confidence

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None):
        if z is None:
            z = self.sample_latent(pc.shape[0])
        if self.is_bimanual_v3:
            dir1, dir2, app1, app2, point1, point2, confidence = self.decode(pc, z, self.is_bimanual_v2, is_bimanual_v3=True)
            return dir1, dir2, app1, app2, point1, point2, confidence, z.squeeze()
        else:
            qt, confidence = self.decode(pc, z, self.is_bimanual_v2)
            return qt, confidence, z.squeeze()

    def generate_dense_latents(self, resolution):
        """
        For the VAE sampler we consider dense latents to correspond to those between -2 and 2
        """
        latents = torch.meshgrid(*[
            torch.linspace(-2, 2, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1).to(self.device)


def base_network(pointnet_radius, pointnet_nclusters, scale, in_features, is_dgcnn=False, device='cuda', use_anchor=False):
    if is_dgcnn:
        base_dgcnn = BaseDgcnn(input_feature=in_features, device=device)
        fc_layer = nn.Sequential(nn.Linear(512 * scale, 1024 * scale),
                                nn.BatchNorm1d(1024 * scale), nn.ReLU(True),
                                nn.Linear(1024 * scale, 1024 * scale),
                                nn.BatchNorm1d(1024 * scale), nn.ReLU(True))
        return nn.ModuleList([base_dgcnn, fc_layer])
    elif use_anchor:
        sa1_module = pointnet2.PointnetSAModule(
            npoint=pointnet_nclusters[0],
            radius=pointnet_radius[0],
            nsample=64,
            mlp=[in_features, 64 * scale, 64 * scale, 128 * scale])
        sa2_module = pointnet2.PointnetSAModule(
            npoint=pointnet_nclusters[1], #32
            radius=pointnet_radius[1], #0.04
            nsample=128,
            mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale])

        sa3_module = pointnet2.PointnetSAModule(
            mlp=[256 * scale, 256 * scale, 256 * scale, 512 * scale])

        sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
        fc_layer = nn.Sequential(nn.Linear(512 * scale, 1024 * scale),
                                nn.BatchNorm1d(1024 * scale), nn.ReLU(True),
                                nn.Linear(1024 * scale, 1024 * scale),
                                nn.BatchNorm1d(1024 * scale), nn.ReLU(True))
        
        return nn.ModuleList([sa_modules, fc_layer])
        

    else:
        sa1_module = pointnet2.PointnetSAModule(
            npoint=pointnet_nclusters[0],
            radius=pointnet_radius[0],
            nsample=64,
            mlp=[in_features, 64 * scale, 64 * scale, 128 * scale])
        sa2_module = pointnet2.PointnetSAModule(
            npoint=pointnet_nclusters[1],
            radius=pointnet_radius[1], #0.04
            nsample=128,
            mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale])

        sa3_module = pointnet2.PointnetSAModule(
            mlp=[256 * scale, 256 * scale, 256 * scale, 512 * scale])

        sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
        fc_layer = nn.Sequential(nn.Linear(512 * scale, 1024 * scale),
                                nn.BatchNorm1d(1024 * scale), nn.ReLU(True),
                                nn.Linear(1024 * scale, 1024 * scale),
                                nn.BatchNorm1d(1024 * scale), nn.ReLU(True))
        return nn.ModuleList([sa_modules, fc_layer])


class BaseDgcnn(nn.Module):
    def __init__(self, opt=None, input_feature=None, device='cuda'):
        super().__init__()
        self.opt = opt
        self.input_feature = input_feature
        self.device = device
        # self.k = self.opt.config['base']['model']['dgcnn']['k']
        self.k = 20
        self.transform_net = Transform_Net(opt)

        # self.emb_dims = self.opt.config['base']['model']['dgcnn']['emb_dims']
        self.emb_dims = 512
        # self.dropout = self.opt.config['base']['model']['dgcnn']['dropout']

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6+self.input_feature, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        
            


    def forward(self, x, rot_feat):

        x = x.transpose(2, 1)
        batch_size = x.size(0)
        num_points = x.size(2)


        x0 = get_graph_feature(x, k=self.k, device=self.device)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x, t = x.double(), t.double()
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k,device=self.device)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        #* concat x and grasp rotation matrix
        rot_feat = rot_feat.unsqueeze(-1).expand(-1,-1,-1,self.k)
        x = torch.cat((x, rot_feat), dim=1)     # (batch_size, 6+16, num_points)

        x = x.float()
        x = self.conv1(x)                       # (batch_size, 22, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, device=self.device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        #* num points to 1
        x = x.max(dim=-1, keepdim=True)[0]     # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        return x

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False, device=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class Transform_Net(nn.Module):
    def __init__(self, opt=None):
        super(Transform_Net, self).__init__()
        self.opt = opt
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        x = x.float()
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x