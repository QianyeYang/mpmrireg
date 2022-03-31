import os
from config.global_train_config import config
import torch

if not config.using_HPC:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{config.gpu}"

if __name__ == "__main__":

    if config.project == "mpmrireg":
        from src.model.archs.mpmrireg import mpmrireg
        model = mpmrireg(config)
    else:
        raise NotImplementedError

    if config.continue_epoch != '-1':
        model.load_epoch(config.continue_epoch)

    model.train()
    print('Optimization done.')
