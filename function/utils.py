import os

import torch


def _get_psnr_from_filename(name):
    return float(name.replace('.pth', '').split('_')[1])


def save_if_topk(model, ep, psnr, weight_path, topk):
    ckpts = list(os.listdir(weight_path))

    if len(ckpts) < topk:
        torch.save(model.state_dict(),
                   os.path.join(weight_path, f"{ep}_{round(psnr, 4)}.pth"))
    else:
        min_ckpt = min(ckpts, key=_get_psnr_from_filename)
        min_psnr = _get_psnr_from_filename(min_ckpt)
        if psnr > min_psnr:
            os.remove(os.path.join(weight_path, min_ckpt))
            torch.save(model.state_dict(),
                       os.path.join(weight_path, f"{ep}_{round(psnr, 4)}.pth"))
    return
