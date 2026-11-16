"""Pre-process the 50k ImageNet val images once with the exact timm eval transform up to
ToTensor (Resize 248 bicubic + CenterCrop 224) and store them as a uint8 array, so every
later evaluation is GPU-bound instead of JPEG-decoding-bound (see data.CachedVal)."""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import data
from common import load_backbone


def to_uint8_chw(img):
    return torch.from_numpy(np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).copy())


def check(tf, idx=(0, 12345, 49999)):
    """Equivalence of the cache with the on-the-fly eval transform."""
    cache = data.CachedVal(batch_size=1)
    files = data.ImageNetFiles('val', tf)
    for j in idx:
        ref = files[j][0]
        x = torch.from_numpy(np.array(cache.x[j])).to(cache.device)[None]
        got = ((x.float() - cache.mean) / cache.std)[0].cpu()
        assert files[j][1] == cache.y[j].item()
        print(f'image {j}: max |cached - on-the-fly| = {(ref - got).abs().max().item():.2e}')


def main():
    vit = load_backbone()
    tf = data.eval_transform(vit)
    if os.path.exists(data.VAL_CACHE + '_y.npy'):
        return check(tf)
    prefix = transforms.Compose(list(tf.transforms[:-2]) + [to_uint8_chw])  # drop ToTensor + Normalize
    ds = data.ImageNetFiles('val', prefix)
    x = np.lib.format.open_memmap(data.VAL_CACHE + '_x.npy', mode='w+', dtype=np.uint8, shape=(len(ds), 3, 224, 224))
    i = 0
    for img, _ in DataLoader(ds, batch_size=250, num_workers=14):
        x[i:i + len(img)] = img.numpy()
        i += len(img)
    x.flush()
    np.save(data.VAL_CACHE + '_y.npy', np.array(ds.targets, dtype=np.int64))
    print(f'cached {i} images')
    check(tf)


if __name__ == '__main__':
    main()
