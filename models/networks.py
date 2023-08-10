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
        net = GraspSamplerVAE(opt.model_scale, opt.pointnet_radius,
                              opt.pointnet_nclusters, opt.latent_size, device, opt.is_bimanual_v2, opt.is_dgcnn)
    elif arch == 'gan':
        net = GraspSamplerGAN(opt.model_scale, opt.pointnet_radius,
                              opt.pointnet_nclusters, opt.latent_size, device)
    elif arch == 'evaluator':
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
                       num_input_features, is_bimanual_v2=False, is_dgcnn=False):
        # The number of input features for the decoder is 3+latent space where 3
        # represents the x, y, z position of the point-cloud
        if not is_bimanual_v2:
            if is_dgcnn:
                self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                            model_scale, num_input_features, is_dgcnn=True)
                self.q = nn.Linear(model_scale * 1024, 4)
                self.t = nn.Linear(model_scale * 1024, 3)
                self.confidence = nn.Linear(model_scale * 1024, 1)
            else:
                self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                            model_scale, num_input_features)
                self.q = nn.Linear(model_scale * 1024, 4)
                self.t = nn.Linear(model_scale * 1024, 3)
                self.confidence = nn.Linear(model_scale * 1024, 1)
        else:
            self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                        model_scale, num_input_features)
            self.q1 = nn.Linear(model_scale * 1024, 4)
            self.t1 = nn.Linear(model_scale * 1024, 3)
            self.q2 = nn.Linear(model_scale * 1024, 4)
            self.t2 = nn.Linear(model_scale * 1024, 3)
            self.confidence = nn.Linear(model_scale * 1024, 1)
            
    def decode(self, xyz, z, is_bimanual_v2=False, is_dgcnn=False):
        if not is_bimanual_v2:
            if is_dgcnn:
                xyz_features = z.unsqueeze(1).expand(-1, xyz.shape[1], -1)
                xyz_features = torch.transpose(xyz_features, 1, 2).contiguous()
                xyz_features = self.decoder[0](xyz, xyz_features)
                x = self.decoder[1](xyz_features.squeeze(-1))
            else:
                xyz_features = self.concatenate_z_with_pc(xyz,z).transpose(-1,1).contiguous()
                for module in self.decoder[0]:
                    xyz, xyz_features = module(xyz, xyz_features)
                x = self.decoder[1](xyz_features.squeeze(-1))
            
            predicted_qt = torch.cat(
                (F.normalize(self.q(x), p=2, dim=-1), self.t(x)), -1)
            return predicted_qt, torch.sigmoid(self.confidence(x)).squeeze()
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
        z.unsqueeze_(1)
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
                 is_dgcnn=False):
        super(GraspSamplerVAE, self).__init__(latent_size, device)
        self.device = device

        self.is_bimanual_v2 = is_bimanual_v2
        self.is_dgcnn = is_dgcnn
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters, self.is_dgcnn)

        if self.is_dgcnn:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size, self.is_bimanual_v2, is_dgcnn=True)
        else:
            self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                                latent_size + 3, self.is_bimanual_v2)
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
        elif self.is_dgcnn:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters, model_scale, 16, is_dgcnn=True, device=self.device)
        else:
            self.encoder = base_network(pointnet_radius, pointnet_nclusters,
                                        model_scale, 19)

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
                xyz, xyz_features = module(xyz, xyz_features)
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
        start = time()
        qt, confidence = self.decode(pc, z, self.is_bimanual_v2, self.is_dgcnn)
        end = time()
        # print('decode time', end - start)
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
        mu, _ = self.bottleneck(z)
        qt, confidence = self.decode(pc, mu, self.is_bimanual_v2, self.is_dgcnn)
            
        return qt, confidence

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None):
        if z is None:
            z = self.sample_latent(pc.shape[0])
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
                 device="cpu"):
        super(GraspSamplerGAN, self).__init__(latent_size, device)
        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                            latent_size + 3)

    def sample_latent(self, batch_size, device=None):
        return torch.rand(batch_size, self.latent_size).to(device)

    def forward(self, pc, grasps=None, train=True):
        z = self.sample_latent(pc.shape[0], device=pc.device)
        return self.decode(pc, z)

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


def base_network(pointnet_radius, pointnet_nclusters, scale, in_features, is_dgcnn=False, device='cuda'):
    if is_dgcnn:
        base_dgcnn = BaseDgcnn(input_feature=in_features, device=device)
        fc_layer = nn.Sequential(nn.Linear(512 * scale, 1024 * scale),
                                nn.BatchNorm1d(1024 * scale), nn.ReLU(True),
                                nn.Linear(1024 * scale, 1024 * scale),
                                nn.BatchNorm1d(1024 * scale), nn.ReLU(True))
        return nn.ModuleList([base_dgcnn, fc_layer])
    else:
    
        sa1_module = pointnet2.PointnetSAModule(
            npoint=pointnet_nclusters,
            radius=pointnet_radius,
            nsample=64,
            mlp=[in_features, 64 * scale, 64 * scale, 128 * scale])
        sa2_module = pointnet2.PointnetSAModule(
            npoint=32,
            radius=0.04,
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