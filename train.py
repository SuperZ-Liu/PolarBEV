import os
import time
import socket
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from polarbev.config import get_parser, get_cfg
from polarbev.data import prepare_dataloaders
from polarbev.trainer import TrainingModule


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    pl.seed_everything(2022)
    
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    trainloader, valloader = prepare_dataloaders(cfg)
    model = TrainingModule(cfg.convert_to_dict())

    if cfg.PRETRAINED.LOAD_WEIGHTS:
        # Load single-image instance segmentation model.
        pretrained_model_weights = torch.load(
            os.path.join('', cfg.PRETRAINED.PATH), map_location='cpu'
        )['state_dict']

        model.load_state_dict(pretrained_model_weights, strict=False)
        print(f'Loaded single-image model weights from {cfg.PRETRAINED.PATH}')

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model'),
    ]
    trainer = pl.Trainer(
        gpus=cfg.GPUS,
        callbacks=callbacks,
        accelerator='ddp',
        precision=cfg.PRECISION,
        sync_batchnorm=True,
        gradient_clip_val=cfg.GRAD_NORM_CLIP,
        max_steps=cfg.STEPS,
        weights_summary='full',
        logger=tb_logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        plugins=DDPPlugin(find_unused_parameters=True),
        profiler='simple',
    )
    trainer.fit(model, trainloader, valloader)


if __name__ == "__main__":
    main()
