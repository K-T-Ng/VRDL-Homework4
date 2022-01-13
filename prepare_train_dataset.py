import os
import shutil

from tqdm import tqdm
from PIL import Image


def renew_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return


def produce_pairs(HR, scale):
    W, H = HR.width, HR.height
    sW, sH = W - (W % scale), H - (H % scale)
    HR = HR.crop((0, 0, sW, sH))

    tW, tH = int(sW // scale), int(sH // scale)
    LR = HR.resize((tW, tH), Image.BICUBIC)
    return LR, HR

if __name__ == '__main__':
    rawloc = os.path.join('datasets', 'training_hr_images')

    TrHR = os.path.join('datasets', 'TrHR')
    TrLR = os.path.join('datasets', 'TrLR')
    renew_dir(TrHR)
    renew_dir(TrLR)

    scaleList = ['2x', '3x', '4x']
    for scale in scaleList:
        os.makedirs(os.path.join(TrHR, scale))
        os.makedirs(os.path.join(TrLR, scale))

    nameList = os.listdir(rawloc)
    for name in tqdm(nameList):
        raw = Image.open(os.path.join(rawloc, name))
        for scale, scaleFolder in enumerate(scaleList, start=2):
            LR, HR = produce_pairs(raw, scale)
            LR.save(os.path.join(TrLR, scaleFolder, name))
            HR.save(os.path.join(TrHR, scaleFolder, name))
