#!/usr/bin/env python
# coding: utf-8

import argparse
import ruamel_yaml as yaml
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from defocus.model import Model
from defocus.utilities import Bunch
import defocus.callbacks as Callbacks

parser = argparse.ArgumentParser(description='It is time for more... experiments.')
parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Experiment configuration')
parser.add_argument('--dryrun', default=None, action='store_true', help='do not upload to wandb')

def main(args):
    if args.training.seed:       
        pl.seed_everything(args.training.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

        
    wandb_logger = WandbLogger(name='dump_defocus_multi_gpu_fp32_no_track_bs16',
                               project='defocus',
                               log_model=False,
                               save_dir='../all_logs/defocus')

    callbacks = [LearningRateMonitor(logging_interval='step'),
                 ModelCheckpoint(dirpath='../all_models/defocus',
                                 monitor='val_PSNR',
                                 save_top_k=5,
                                 save_last=True,
                                 mode='max',
                                )
                ]
    if hasattr(Callbacks, args.model.callbacks):
        callbacks += [getattr(Callbacks, args.model.callbacks)()]
    trainer = pl.Trainer(gpus=args.training.gpus,
                        accelerator='ddp',
                        callbacks=callbacks,
                        logger=wandb_logger,
                        log_every_n_steps=1,
                        max_epochs=args.training.max_epochs,
                        precision=args.training.precision,
    #                      limit_train_batches=10,
    #                      limit_val_batches=10,
                        )
    model = Model(args)
    trainer.fit(model)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"  
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
        # Bunch is for **recursively** creating a Namespace from dict object
        bunch = Bunch(config)
        # but lightning must have a Namespace object, so convert to Namespace back again
        args = argparse.Namespace(**vars(bunch))
    args = argparse.Namespace(**vars(bunch))
    main(args)

    