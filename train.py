import os
from config import args
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

if __name__ == "__main__":
    if args.model == "origin":
        model = archs.mpMRIRegUnsupervise(args)
    elif args.model == "weakly":
        from src.model.archs.mpMRIRegWeakSupervise import weakSuperVisionMpMRIReg
        model = weakSuperVisionMpMRIReg(args)
    elif args.model == "multi-task":
        pass
    elif args.model == "joint3":
        model = archs.joint3(args)
    elif args.model == "test-affine-optim":
        model = archs.test_affine_optim(args)
    else:
        raise NotImplementedError
    if args.continue_epoch != -1:
        print(f'load from checkpoints {args.continue_epoch}')
        model.load_ckpt(args.continue_epoch)
    model.train()
    print('Optimization done.')