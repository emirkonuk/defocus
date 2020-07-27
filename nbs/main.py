'''
import argparse
import importlib
import torch
import torch.nn as nn
from defocus.model import Model
import random
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist

import time

parser = argparse.ArgumentParser(description='It is time for more... experiments.')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=6, help='the number of dataloader workers')
parser.add_argument('--distributed', action='store_true', help='blurb')
parser.add_argument('--world_size', default=1, type=int, help='number of gpus')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8842', type=str, help='single machine, change port if needed')

def main():
    args = parser.parse_args()

    if args.world_size>1:
        args.distributed = True
        mp.spawn(main_worker, nprocs=args.world_size, args=(args,))
    else:
        device_id = 0
        main_worker(device_id, args)



def main_worker(device_id, args, seed=0):
    # set the seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    world_size = args.world_size
    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=world_size, rank=device_id)

    model_name = 'MSResNet'
    architecture = importlib.import_module('defocus.architecture.' + model_name)
    training = importlib.import_module('defocus.trainers.' + model_name)

    G = architecture.Generator()
    D = architecture.Discriminator()
    model = Model()
    model.G = G
    model.D = D
    # model.use_perceptual()
    model.set_G_optimizer('AdamP')
    model.set_D_optimizer('AdamP')
    model.set_resconstruction_loss(loss_functions=[nn.L1Loss()], 
                               weights=[1.0])
    model.set_adversarial_loss(loss_functions=[nn.BCEWithLogitsLoss()],
                           weights=[1.0])

    data = importlib.import_module('defocus.data.' + model_name)
    train_dataset = data.Dataset(root_folder='/storage/projects/all_datasets/GOPRO/train/', 
                                image_pair_list='/storage/projects/all_datasets/GOPRO/train/train_image_pair_list.txt',
                                )
    validation_dataset = data.Dataset(root_folder='/storage/projects/all_datasets/GOPRO/train/', 
                                  image_pair_list='/storage/projects/all_datasets/GOPRO/train/val_image_pair_list.txt',
                                 )

    trainer = training.Trainer(model, train_dataset, validation_dataset, 
                               batch_size=args.batch_size, 
                               num_workers=args.num_workers, 
                               device_id=device_id,
                               world_size=world_size)
    
    for epoch in range(100):
        start = time.time()
        trainer.train(epoch)
        end = time.time()
        if device_id==0:
            print('Epoch {}, took {} min'.format(epoch, (end - start)/60))
'''

import argparse
from defocus.lightning import MSResNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
from pytorch_lightning.callbacks import ModelCheckpoint

def main(args):
    pl.seed_everything(42)
    experiment_name = '_'.join([args.model_name, str(args.batch_size), str(args.lr), 'nightly'])
    wandb_logger = WandbLogger(name=experiment_name,
                               project='defocus',
                               log_model=True,
                               save_dir='lightning_logs')

    '''
    These checkpoint stuff is because lightning is messy without the 
    default settings. They are just bookkeeping and saving models.
    Disregard them.
    '''
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/596
    class ModelCheckpointAtEpochEnd(pl.Callback):
        def on_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            metrics['epoch'] = trainer.current_epoch

    checkpoint_folder = './lightning_logs/checkpoints/'
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_folder + '{epoch}',
                                          save_top_k=-1,
                                          verbose=True,
                                          monitor='epoch',
                                          mode='max',
                                          prefix='',
                                          )

    gan_model = MSResNet.GAN(args)
    trainer = pl.Trainer(gpus=2, 
                         fast_dev_run=False,
                         distributed_backend='ddp',
                         max_epochs=100,
                         logger=wandb_logger,
                         checkpoint_callback=checkpoint_callback,
                         callbacks=[ModelCheckpointAtEpochEnd()],
                         precision=16 if args.fp16 else 32,
                         )    
    trainer.fit(gan_model) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='It is time for more... experiments.')
    parser.add_argument('--model_name', type=str, default='MSResNet', help='model name')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of dataloader workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='ADAM betas')
    parser.add_argument('--adv_loss', nargs='+', default=('BCEWithLogitsLoss', '1.0'), action='store', 
                        help='Adversarial loss function(s) and weighting, e.g. BCEWithLogitsLoss 0.5 MSELoss 0.5')
    parser.add_argument('--rec_loss', nargs='+', default=('MSELoss', '1.0'), action='store',
                        help='Reconstruction loss function(s) and weighting, e.g. L1Loss 0.5 MSELoss 0.5')
    parser.add_argument('--per_loss', nargs='+', action='store', 
                        help='Perceptual loss function(s) and weighting, e.g. L1Loss 0.5 MSELoss 0.5')
    parser.add_argument('--fp16', action='store_true',
                        help='Mixed precision')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use. currently Adam or AdamP')
    parser.add_argument('--milestones', type=int, nargs='+', default=[500, 750, 900], help='learning rate decay per N epochs')
    parser.add_argument('--root_folder', type=str, default='/storage/projects/all_datasets/GOPRO/train/', help='root folder')
    parser.add_argument('--image_pair_list', type=str, default='/storage/projects/all_datasets/GOPRO/train/train_image_pair_list.txt', help='image list')
    args = parser.parse_args()
    main(args)