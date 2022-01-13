import os
import random

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from function.Dataset import patchDataset
from function.TrainFunction import train, valid
from function.utils import save_if_topk
from model.DRLN import DRLN

if __name__ == '__main__':
    # set up path
    TrHR2x = os.path.join('datasets', 'TrHR', '2x')
    TrHR3x = os.path.join('datasets', 'TrHR', '3x')
    TrHR4x = os.path.join('datasets', 'TrHR', '4x')

    TrLR2x = os.path.join('datasets', 'TrLR', '2x')
    TrLR3x = os.path.join('datasets', 'TrLR', '3x')
    TrLR4x = os.path.join('datasets', 'TrLR', '4x')

    ValHR2x = os.path.join('datasets', 'ValHR', '2x')
    ValHR3x = os.path.join('datasets', 'ValHR', '3x')
    ValHR4x = os.path.join('datasets', 'ValHR', '4x')

    ValLR2x = os.path.join('datasets', 'ValLR', '2x')
    ValLR3x = os.path.join('datasets', 'ValLR', '3x')
    ValLR4x = os.path.join('datasets', 'ValLR', '4x')

    saveLoc2x = os.path.join('datasets', 'Valid', '2x')
    saveLoc3x = os.path.join('datasets', 'Valid', '3x')
    saveLoc4x = os.path.join('datasets', 'Valid', '4x')

    weightLoc = os.path.join('weights')

    # parameters
    epochs = 1500
    batch_size = 16

    scale = 3
    patch_size = 32

    betas = (0.9, 0.999)
    init_lr = 1e-4
    weight_decay = 5e-4
    eps = 1e-8

    step_size = 250
    gamma = 0.5
    save_topk = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    model = DRLN(scale).to(device)

    # get train dataloader
    TrDs2x = patchDataset(LRLoc=TrLR2x, HRLoc=TrHR2x, scale=2, patch_size=48)
    TrDs3x = patchDataset(LRLoc=TrLR3x, HRLoc=TrHR3x, scale=3, patch_size=48)
    TrDs4x = patchDataset(LRLoc=TrLR4x, HRLoc=TrHR4x, scale=4, patch_size=48)

    Loader2x = DataLoader(TrDs2x, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    Loader3x = DataLoader(TrDs3x, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    Loader4x = DataLoader(TrDs4x, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)

    # loss function
    loss_fn = nn.L1Loss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=init_lr, betas=betas,
                           weight_decay=weight_decay, eps=eps)

    # scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                    gamma=gamma)

    for ep in range(1, epochs+1):
        print(f"Epoch: {ep} / {epochs}")

        scale = ep % 3 + 2  # 2, 3 or 4
        if scale == 2:
            Loader = Loader2x
        elif scale == 3:
            Loader = Loader3x
        else:
            Loader = Loader4x
        print(f"Dealing with {scale}x")
        model.change_scale(scale)
        train(model, Loader, loss_fn, optimizer, device)

        scheduler.step()

        model.change_scale(2)
        psnr2x = valid(model, loss_fn, device, ValHR2x, ValLR2x, saveLoc2x)

        model.change_scale(3)
        psnr3x = valid(model, loss_fn, device, ValHR3x, ValLR3x, saveLoc3x)

        model.change_scale(4)
        psnr4x = valid(model, loss_fn, device, ValHR4x, ValLR4x, saveLoc4x)

        save_if_topk(model, ep, psnr3x, weightLoc, save_topk)
