import os
from tqdm import tqdm

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from function.AverageMeter import AvgMeter


def init_result():
    result = {
        'loss': AvgMeter(),
        'psnr': AvgMeter()
    }
    return result


def train(model, Loader, loss_fn, optimizer, device):
    model.train()
    result = init_result()
    bar = tqdm(Loader)
    for LR, HR in bar:
        LR = LR.to(device)
        HR = HR.to(device)

        optimizer.zero_grad()
        SR = model(LR)
        loss = loss_fn(SR, HR)
        loss.backward()
        optimizer.step()

        psnr = 10. * torch.log10(1. / torch.mean((SR - HR) ** 2))
        result['loss'].update(loss.item(), LR.size(0))
        result['psnr'].update(psnr.item(), LR.size(0))

        bar.set_postfix({key: val.item() for key, val in result.items()})


def valid(model, loss_fn, device, ValHRLoc, ValLRLoc, saveLoc):
    model.eval()
    result = init_result()
    nameList = tqdm(os.listdir(ValHRLoc))
    for name in nameList:
        LR = read_image(os.path.join(ValLRLoc, name)) / 255.
        HR = read_image(os.path.join(ValHRLoc, name)) / 255.

        LR = LR.unsqueeze(0).to(device)
        HR = HR.unsqueeze(0).to(device)

        with torch.no_grad():
            SR = model(LR)
            loss = loss_fn(SR, HR)
            psnr = 10. * torch.log10(1. / torch.mean((SR - HR) ** 2))

        result['loss'].update(loss.item(), LR.size(0))
        result['psnr'].update(psnr.item(), LR.size(0))
        save_image(SR, os.path.join(saveLoc, name))

        nameList.set_postfix({key: val.item() for key, val in result.items()})
    return result['psnr'].item()
