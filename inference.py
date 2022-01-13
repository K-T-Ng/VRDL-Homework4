import os

import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm

from model.DRLN import DRLN


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


def self_Ensemble(model, LR, device, forward, backward):
    '''
    LR (torch.tensor): low-resolution image with shape (3, H, W)
    '''
    with torch.no_grad():
        SR = LR.to(device)
        SR = forward(SR)
        SR = SR.unsqueeze(0)
        SR = model(SR)
        SR = SR.squeeze()
        SR = backward(SR)
    return SR

if __name__ == '__main__':
    # set up path
    testLoc = os.path.join('datasets', 'testing_lr_images')
    resultLoc = os.path.join('answer')
    ckpt = os.path.join('weights', '1474_31.5458.pth')
    testList = os.listdir(testLoc)

    # read model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DRLN(scale=3)
    model.load_state_dict(torch.load(ckpt))
    model.eval().to(device)

    # Ensemble transform
    Ensemble_Tform = [
        {'forward': Identity, 'backward': Identity},
        {'forward': FlipLR, 'backward': FlipLR},
        {'forward': FlipUD, 'backward': FlipUD},
        {'forward': Rot90, 'backward': Rot270},
        {'forward': Rot180, 'backward': Rot180},
        {'forward': Rot270, 'backward': Rot90},
    ]

    # inference with self-ensemble
    for imgname in tqdm(testList):
        imgID = imgname.split('.')[0]

        LR = read_image(os.path.join(testLoc, imgname))
        LR = LR / 255.

        H = LR.size(1)
        W = LR.size(2)

        SR = torch.zeros(3, 3*H, 3*W).to(device)

        for tform in Ensemble_Tform:
            sr = self_Ensemble(model, LR, device,
                               tform['forward'], tform['backward'])
            SR += sr

        SR = SR / len(Ensemble_Tform)
        save_image(SR, os.path.join(resultLoc, imgID+'_pred.png'))
