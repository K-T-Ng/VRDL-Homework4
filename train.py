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
    TrHR = os.path.join('datasets', 'TrHR', '3x')
    TrLR = os.path.join('datasets', 'TrLR', '3x')
    ValHR = os.path.join('datasets', 'ValHR', '3x')
    ValLR = os.path.join('datasets', 'ValLR', '3x')
    saveLoc = os.path.join('datasets', 'Valid')
    weightLoc = os.path.join('weights')

    # parameters
    epochs = 400
    batch_size = 8

    scale = 3
    patch_size = 64

    betas = (0.9, 0.999)
    init_lr = 1e-4
    weight_decay = 0
    eps = 1e-8

    step_size = 50
    gamma = 0.5
    save_topk = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    model = DRLN(scale).to(device)

    # get train dataloader
    TrDs = patchDataset(LRLoc=TrLR, HRLoc=TrHR, scale=scale,
                        patch_size=patch_size)
    TrLoader = DataLoader(TrDs, batch_size=batch_size, shuffle=True,
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
        train(model, TrLoader, loss_fn, optimizer, device)
        scheduler.step()
        psnr = valid(model, loss_fn, device, ValHR, ValLR, saveLoc)
        save_if_topk(model, ep, psnr, weightLoc, save_topk)
