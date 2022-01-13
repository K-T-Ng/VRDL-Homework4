import os
import random

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms


class patchDataset(Dataset):
    def __init__(self, LRLoc, HRLoc, scale, patch_size):
        self.LRLoc = LRLoc
        self.HRLoc = HRLoc
        self.nameList = os.listdir(LRLoc)
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.nameList)

    def __getitem__(self, index):
        name = self.nameList[index]
        LR = read_image(os.path.join(self.LRLoc, name))
        HR = read_image(os.path.join(self.HRLoc, name))

        LR, HR = get_patch(LR, HR, self.patch_size, self.scale)
        LR, HR = random_augment(LR, HR)

        LR = LR / 255.
        HR = HR / 255.
        return LR, HR


def pad(img, imgW, imgH, cropW, cropH, padW, padH):
    padding_ltrb = [
        (cropW - imgW) // 2 if padW else 0,
        (cropH - imgH) // 2 if padH else 0,
        (cropW - imgW + 1) // 2 if padW else 0,
        (cropH - imgH + 1) // 2 if padH else 0
    ]
    return TF.pad(img, padding_ltrb, fill=0)


def get_patch(LR, HR, patch_size=64, scale=3):
    lH, hH = LR.size(1), HR.size(1)
    lW, hW = LR.size(2), HR.size(2)

    p_in = patch_size
    p_out = patch_size * scale

    padH = p_in > lH
    padW = p_in > lW

    # If LR is not large enough, pad.
    if padH or padW:
        LR = pad(LR, lW, lH, p_in, p_in, padW, padH)
        HR = pad(HR, hW, hH, p_out, p_out, padW, padH)

    # random crop
    lH, hH = LR.size(1), HR.size(1)
    lW, hW = LR.size(2), HR.size(2)
    lx = random.randrange(0, lW - p_in + 1)
    ly = random.randrange(0, lH - p_in + 1)
    hx = lx * scale
    hy = ly * scale
    LR = LR[:, ly:ly+p_in, lx:lx+p_in]
    HR = HR[:, hy:hy+p_out, hx:hx+p_out]

    return LR, HR


def random_augment(LR, HR):
    ID = random.randint(0, 5)
    if ID == 0:
        aug = Identity
    elif ID == 1:
        aug = FlipLR
    elif ID == 2:
        aug = FlipUD
    elif ID == 3:
        aug = Rot90
    elif ID == 4:
        aug = Rot180
    elif ID == 5:
        aug = Rot270
    LR = aug(LR)
    HR = aug(HR)
    return LR, HR


def Identity(x):
    return x


def FlipLR(x):
    return TF.hflip(x)


def FlipUD(x):
    return TF.vflip(x)


def Rot90(x):
    return torch.rot90(x, k=1, dims=[1, 2])


def Rot180(x):
    return torch.rot90(x, k=2, dims=[1, 2])


def Rot270(x):
    return torch.rot90(x, k=3, dims=[1, 2])
