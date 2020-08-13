# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/model_without_lightning.ipynb (unless otherwise specified).

__all__ = ['Model']

# Cell
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from adamp import AdamP
from torch.nn.parallel import DistributedDataParallel, DataParallel

# Cell
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.G = None
        self.D = None
        self.P = None
        self.G_optimizer = None
        self.D_optimizer = None
        self.reconstruction_loss = None
        self.adversarial_loss = None
        self.perceptual_loss = None

    def use_perceptual(self, after_activation=False):
        # this is from DeblurGANv2
        # TODO: ESRGAN's perceptual loss version
        conv_3_3_layer = 14
        cnn = torchvision.models.vgg19(pretrained=True).features
        perceptual = nn.Sequential()
        perceptual = perceptual.eval()
        for i, layer in enumerate(list(cnn)):
            perceptual.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        self.P = perceptual

    def set_resconstruction_loss(self, loss_functions=[nn.MSELoss], weights=[1.0]):
        def weighted_loss(input_, target):
            total = 0
            for (func, weight) in zip(loss_functions, weights):
                total += func(input_, target)*weight
            return total
        self.reconstruction_loss = weighted_loss

    def set_adversarial_loss(self, loss_functions=[nn.BCEWithLogitsLoss], weights=[1.0]):
        def weighted_loss(input_, target):
            total = 0
            for (func, weight) in zip(loss_functions, weights):
                total += func(input_, target)*weight
            return total
        self.adversarial_loss = weighted_loss

    def set_perceptual_loss(self, loss_functions=[nn.L1Loss], weights=[1.0]):
        def weighted_loss(input_, target):
            total = 0
            for (func, weight) in zip(loss_functions, weights):
                total += func(input_, target)*weight
            return total
        self.perceptual_loss = weighted_loss

    def set_G_optimizer(self, optimizer='AdamP', lr=1e-4, betas=(0.9, 0.999), weight_decay=0, nesterov=False):
        # note that lucidrains uses betas=(0.5, 0.9) for stylegan
        # https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py#L565

        if optimizer=='Adam':
            self.G_optimizer = Adam(self.G.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer=='AdamP':
            self.G_optimizer = AdamP(self.G.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, nesterov=nesterov)
        else:
            #TODO: other optimizers, maybe from the torch_optimizers package
            raise NotImplementedError('nope')

    def set_D_optimizer(self, optimizer='AdamP', lr=1e-4, betas=(0.9, 0.999), weight_decay=0, nesterov=False):
        # note that lucidrains uses betas=(0.5, 0.9) for stylegan
        # https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py#L565

        if optimizer=='Adam':
            self.D_optimizer = Adam(self.D.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer=='AdamP':
            self.D_optimizer = AdamP(self.D.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, nesterov=nesterov)
        else:
            #TODO: other optimizers, maybe from the torch_optimizers package
            raise NotImplementedError('nope')

    def save(self, model_path):
        if isinstance(self.G, DataParallel) or isinstance(self.G, DistributedDataParallel):
            G_state_dict = self.G.module.state_dict()
            D_state_dict = self.D.module.state_dict()
        else:
            G_state_dict = self.G.state_dict()
            D_state_dict = self.D.state_dict()
        torch.save({'G':G_state_dict,
                    'D':D_state_dict,
                    'optimizer_G': self.G_optimizer.state_dict(),
                    'optimizer_D': self.D_optimizer.state_dict()},
                   model_path)

    def load(self, model_path, isStrict=False, map_location='cpu'):
        # don't care about distributed here
        # trainer should take care of it after loading the parameters
        checkpoint = torch.load(model_path, map_location=map_location)
        self.G.load_state_dict(checkpoint['G'], strict=isStrict)
        self.D.load_state_dict(checkpoint['D'], strict=isStrict)
        self.G_optimizer.load_state_dict(checkpoint['optimizer_G'])
        self.D_optimizer.load_state_dict(checkpoint['optimizer_D'])