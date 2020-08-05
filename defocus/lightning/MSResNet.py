# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/lightning_trial.ipynb (unless otherwise specified).

__all__ = ['GAN']

# Cell
import importlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
from collections import OrderedDict
from adamp import AdamP
import wandb

class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()

        self.hparams = hparams
        architecture = importlib.import_module('defocus.architecture.' + self.hparams.model_name)
        self.G = architecture.Generator()
        self.D = architecture.Discriminator()

        data = importlib.import_module('defocus.data.' + self.hparams.model_name)
        self.Dataset = data.Dataset

        self.adversarial_loss = self.set_weighted_loss(loss_functions=[getattr(nn, funcname)() for funcname in self.hparams.adv_loss[0::2]],
                                                  weights=[float(weight) for weight in self.hparams.adv_loss[1::2]],
                                                 )
        self.reconstruction_loss = self.set_weighted_loss(loss_functions=[getattr(nn, funcname)() for funcname in self.hparams.rec_loss[0::2]],
                                                  weights=[float(weight) for weight in self.hparams.rec_loss[1::2]],
                                                 )

        if self.hparams.per_loss is not None:
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
            self.perceptual_loss = self.set_weighted_loss(loss_functions=[getattr(nn, funcname)() for funcname in self.hparams.per_loss[0::2]],
                                                      weights=[float(weight) for weight in self.hparams.per_loss[1::2]],
                                                     )

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None


        self.reduced_loss_g_adversarial = 0
        self.reduced_loss_d = 0

    def set_weighted_loss(self, loss_functions=[nn.BCEWithLogitsLoss], weights=[1.0]):
        def weighted_loss(input_, target):
            total = 0
            for (func, weight) in zip(loss_functions, weights):
                total += func(input_, target)*weight
            return total
        return weighted_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = self.hparams.betas
        optimizer = self.hparams.optimizer

        if optimizer == 'AdamP':
            opt_g = AdamP(self.G.parameters(), lr=lr, betas=betas, weight_decay=0, nesterov=False)
            opt_d = AdamP(self.D.parameters(), lr=lr, betas=betas, weight_decay=0, nesterov=False)
        else:
            opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=betas, weight_decay=0)
            opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=betas, weight_decay=0)

        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=[500,750,900], gamma=0.5)
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[500,750,900], gamma=0.5)
        return [opt_d, opt_g], []#[scheduler_d, scheduler_g]

    def train_dataloader(self):
        train_dataset = self.Dataset(root_folder=self.hparams.root_folder,
                                     image_pair_list=self.hparams.image_pair_list,
                                    )
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

#     def val_dataloader(self):
#         val_dataset = self.Dataset(root_folder='/storage/projects/all_datasets/GOPRO/train/',
#                                           image_pair_list='/storage/projects/all_datasets/GOPRO/train/val_image_pair_list.txt',
#                                          )
#         return DataLoader(val_dataset, batch_size=self.hparams.batch_size)

    def forward(self, input_):
        return self.G(input_)

    def training_step(self, batch, batch_idx, optimizer_idx):
        input_, target = batch
        self.last_imgs = input_

        output = self.G(input_)
        loss_g_rec = 0
        for scaled_output, scaled_target in zip(output, target):
            loss_g_rec += self.reconstruction_loss(scaled_output, scaled_target)
        high_res_output = output[-1]
        high_res_target = target[-1]

        # train discriminator
        if optimizer_idx == 0:
            d_fake = self.D(high_res_output.detach())
            d_real = self.D(high_res_target)

            label_fake = torch.zeros_like(d_fake)
            label_real = torch.ones_like(d_real)
            if self.on_gpu:
                label_fake = label_fake.cuda(high_res_output.device.index)
                label_real = label_real.cuda(high_res_output.device.index)

            loss_d = self.adversarial_loss(d_fake, label_fake) + self.adversarial_loss(d_real, label_real)
            # this is unnecessary but I will use it for flood loss (maybe) and adaptive no-gan
            reduced_loss_d = loss_d.clone()
            torch.distributed.all_reduce_multigpu([reduced_loss_d], op=torch.distributed.ReduceOp.SUM)
            self.reduced_loss_d = reduced_loss_d.item()

            tqdm_dict = {'loss_d': loss_d.item()}
            losses_and_logs = OrderedDict({
                'loss': loss_d,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return losses_and_logs

        # train generator
        if optimizer_idx == 1:
            d_fake_with_gradient = self.D(high_res_output)

            label_real = torch.ones_like(d_fake_with_gradient)
            if self.on_gpu:
                label_real = label_real.cuda(high_res_output.device.index)

            loss_g_adversarial = self.adversarial_loss(d_fake_with_gradient, label_real)
            loss_g = loss_g_adversarial + loss_g_rec

            # again, this is unnecessary but I will use it for flood loss (maybe) and adaptive no-gan
            reduced_loss_g_adversarial = loss_g_adversarial.clone()
            torch.distributed.all_reduce_multigpu([reduced_loss_g_adversarial], op=torch.distributed.ReduceOp.SUM)
            self.reduced_loss_g_adversarial = reduced_loss_g_adversarial.item()



            tqdm_dict = {'loss_g': loss_g.item(), 'loss_g_adversarial': loss_g_adversarial.item()}
            losses_and_logs = OrderedDict({
                'loss': loss_g,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return losses_and_logs


    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None, on_tpu=False, using_native_amp=True, using_lbfgs=False):
        # while updating the discriminator...
        if optimizer_idx == 0:
            if self.reduced_loss_g_adversarial < 2.0:
                self.trainer.scaler.step(optimizer)
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
        # while updating the generator...
        if optimizer_idx == 1:
            if self.reduced_loss_d < 2.0:
                self.trainer.scaler.step(optimizer)
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()

    def on_epoch_end(self):
        pass
        # not working, wandb does not upload the images
#         out = self(self.last_imgs)
#         image = out[-1][0].detach().cpu().numpy().transpose(1,2,0)
#         self.logger.experiment.log({"examples": [wandb.Image(image, caption="output")]})