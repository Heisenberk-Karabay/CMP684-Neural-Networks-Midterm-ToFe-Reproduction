"""Extract downloaded ImageNet-1k parquet shards (Hugging Face ILSVRC/imagenet-1k) to
data/imagenet/<split>/<label>/<name>.JPEG. Already extracted shards are skipped.

usage: python src/prepare_imagenet.py [validation|train ...]
"""
import glob
import os
import sys

import pyarrow.parquet as pq

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'imagenet')
OUT_SPLIT = {'validation': 'val', 'train': 'train'}


def extract(shard, out_root):
    marker = os.path.join(out_root, '.done', os.path.basename(shard))
    if os.path.exists(marker):
        return 0
    pf = pq.ParquetFile(shard)
    n = 0
    stem = os.path.basename(shard).split('.')[0]
    for g in range(pf.num_row_groups):
        tab = pf.read_row_group(g, columns=['image', 'label']).to_pydict()
        for img, label in zip(tab['image'], tab['label']):
            d = os.path.join(out_root, f'{label:04d}')
            os.makedirs(d, exist_ok=True)
            name = os.path.basename(img['path'] or '') or f'{stem}_{n:06d}.JPEG'
            with open(os.path.join(d, name), 'wb') as f:
                f.write(img['bytes'])
            n += 1
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, 'w').close()
    return n


if __name__ == '__main__':
    for split in sys.argv[1:] or ['validation', 'train']:
        out_root = os.path.join(ROOT, OUT_SPLIT[split])
        for shard in sorted(glob.glob(os.path.join(ROOT, f'{split}-*.parquet'))):
            n = extract(shard, out_root)
            if n:
                print(f'{os.path.basename(shard)}: {n} images', flush=True)
        total = sum(len(fs) for _, _, fs in os.walk(out_root)) if os.path.isdir(out_root) else 0
        print(f'{split}: {total} files in {out_root}')
