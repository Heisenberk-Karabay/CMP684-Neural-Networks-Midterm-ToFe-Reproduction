"""ImageNet-1k (extracted parquet shards) and ImageNet subsets (Imagenette / Imagewoof)
with labels mapped to the 1000-class ImageNet head."""
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from timm.data import ImageNetInfo, resolve_data_config, create_transform

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
SUBSETS = {'imagenette': 'imagenette2-320', 'imagewoof': 'imagewoof2-320'}
VAL_CACHE = os.path.join(DATA_ROOT, 'imagenet', 'val224')
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class ImageNetSubset(ImageFolder):
    """ImageFolder whose class folders are WordNet ids; targets become ImageNet-1k indices."""

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        wnid_to_idx = {w: i for i, w in enumerate(ImageNetInfo().label_names())}
        self.samples = [(p, wnid_to_idx[self.classes[t]]) for p, t in self.samples]
        self.imgs = self.samples
        self.targets = [t for _, t in self.samples]


class ImageNetFiles(Dataset):
    """ImageNet-1k extracted by prepare_imagenet.py: data/imagenet/<split>/<label>/*.JPEG."""

    def __init__(self, split, transform=None):
        root = os.path.join(DATA_ROOT, 'imagenet', split)
        self.samples = [(e.path, int(d.name)) for d in sorted(os.scandir(root), key=lambda d: d.name)
                        if d.is_dir() and d.name.isdigit() for e in os.scandir(d.path)]
        self.targets = [t for _, t in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        img = default_loader(path)
        return (self.transform(img) if self.transform else img), target


class CachedVal:
    """Batches of the uint8 val cache written by cache_val.py, normalised on the GPU.
    Identical to ImageNetFiles('val', eval_transform(model)) for DeiT-S."""

    def __init__(self, batch_size=256, device='cuda'):
        self.x = np.load(VAL_CACHE + '_x.npy', mmap_mode='r')
        self.y = torch.from_numpy(np.load(VAL_CACHE + '_y.npy'))
        self.bs, self.device = batch_size, device
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255

    def __len__(self):
        return (len(self.y) + self.bs - 1) // self.bs

    def __iter__(self):
        for i in range(0, len(self.y), self.bs):
            x = torch.from_numpy(np.array(self.x[i:i + self.bs])).to(self.device, non_blocking=True)
            yield (x.float() - self.mean) / self.std, self.y[i:i + self.bs]


def eval_transform(model):
    return create_transform(**resolve_data_config({}, model=model))


def train_transform(model):
    cfg = resolve_data_config({}, model=model)
    return transforms.Compose([
        transforms.RandomResizedCrop(cfg['input_size'][-1], scale=(0.35, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std']),
    ])


def build(names, split, transform):
    sets = [ImageNetSubset(os.path.join(DATA_ROOT, SUBSETS[n], split), transform) for n in names]
    return sets[0] if len(sets) == 1 else ConcatDataset(sets)
