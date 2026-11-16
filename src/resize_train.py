"""Down-scale the extracted ImageNet train images once (shorter side 256, JPEG q=90) into
data/imagenet/train256, so training is GPU-bound rather than JPEG-decode-bound."""
import os
from multiprocessing import Pool

from PIL import Image

import data

SRC = os.path.join(data.DATA_ROOT, 'imagenet', 'train')
DST = os.path.join(data.DATA_ROOT, 'imagenet', 'train256')
SHORT = 256


def resize(job):
    src, dst = job
    if os.path.exists(dst):
        return
    img = Image.open(src).convert('RGB')
    w, h = img.size
    s = SHORT / min(w, h)
    if s < 1:
        img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    img.save(dst, quality=90)


if __name__ == '__main__':
    jobs = []
    for d in os.scandir(SRC):
        if d.is_dir() and d.name.isdigit():
            os.makedirs(os.path.join(DST, d.name), exist_ok=True)
            jobs += [(e.path, os.path.join(DST, d.name, e.name)) for e in os.scandir(d.path)]
    with Pool(15) as pool:
        for i, _ in enumerate(pool.imap_unordered(resize, jobs, chunksize=256)):
            if i % 20000 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
    print(f'done: {len(jobs)} images -> {DST}')
