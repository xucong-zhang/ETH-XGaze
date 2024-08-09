import torch
from trainer import Trainer
from config import get_config
from data_loader import get_train_loader, get_test_loader

import numpy as np

import configparser

def run(config):
    kwargs = {}
    if config.use_gpu:
        # ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        np.random.seed(0)
        kwargs = {'num_workers': config.num_workers}

    # instantiate data loaders
    if config.is_train:
        data_loader = get_train_loader(
            config.data_dir, config.batch_size, is_shuffle=True,
            **kwargs
        )
    else:
        data_loader = get_test_loader(
            config.data_dir, config.batch_size, is_shuffle=False,
            **kwargs
        )
    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()

if __name__ == '__main__':
    config, unparsed = get_config()
    run(config)
