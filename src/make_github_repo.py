"""Build the folder that is pushed to the project's GitHub repository (common.GITHUB_REPO).

usage: python src/make_github_repo.py [--repo DIR]

Everything inside DIR except .git is rebuilt on every run, so the script can be run again after
`git init`. Included: README, report, code, result files and the trained models (largest file
86 MB, below GitHub's 100 MB limit; push with git, the web uploader only takes 25 MB).
Not included: the ImageNet data and the original paper PDF (IEEE copyright, no redistribution).
"""
import argparse
import glob
import json
import os
import shutil

from build_paper import REPORT_PDF
from common import GITHUB_REPO, ROOT, RUNS
from make_figures import load_runs
from make_submission import REQUIREMENTS, copy_results, published_runs

DOI = 'https://doi.org/10.1109/TNNLS.2026.3684383'
ARXIV = 'https://arxiv.org/abs/2507.16260'

GITIGNORE = """# Python
__pycache__/
*.pyc
.venv/

# data, caches and local checkpoints (too large for git)
data/
*.pt
*.pth
*.npy
*.parquet

# published checkpoints (every file is below GitHub's 100 MB limit)
!models/*.pt

# build files
report/ieee/_*.html
report/ieee/_*.pdf
"""


def models_readme(runs):
    lines = ['Trained ToFe checkpoints, one file per run (DeiT-S backbone from timm,',
             'deit_small_patch16_224.fb_in1k). Each file is a dict with "args" (training options),',
             '"keep_avg" (average kept tokens per stage) and "state" (weights of the token selectors and',
             'approximators; tofe_b50_ft3 also contains its fine-tuned backbone).', '',
             'The code finds these files by itself (src/common.py, checkpoint_path):',
             '  from analyze_tofe import load_run; model, keep = load_run("tofe_b50")',
             '  python src/analyze_tofe.py tofe_b50      # evaluates it on the ImageNet validation set', '',
             f'{"run":22s} {"GMACs":>6s} {"top-1":>7s}  kept tokens per stage']
    for n in published_runs(runs):
        r = runs[n]
        lines.append(f"{n:22s} {r['gmacs']:6.2f} {r['top1']:7.2f}  {r['keep_counts']}")
    return '\n'.join(lines) + '\n'


def readme(runs, base):
    b = {(x['method'], str(x['setting'])): x for x in base}
    mean = lambda names: sum(runs[n]['top1'] for n in names) / len(names)
    size = sum(os.path.getsize(os.path.join(RUNS, n, 'model.pt')) for n in published_runs(runs)) / 2**20
    return f"""# Reproducing ToFe: Token Freezing and Reusing for Efficient Vision Transformers

Midterm reading project for **CMP684 Neural Networks** (Hacettepe University), by Hasan Berk Karabay.

In this project I re-implemented the method of the paper below from its text (the authors did
not release code) and checked its main claims on DeiT-S with ImageNet-1k.

> H. Zhang, J. Zhang, C. Hu, and S. Guo, "ToFe: Lagged Token Freezing and Reusing for Efficient
> Vision Transformer Inference," *IEEE Transactions on Neural Networks and Learning Systems*,
> vol. 37, no. 10, pp. 5545-5558, Oct. 2026. [DOI]({DOI}) | [arXiv]({ARXIV})

- Report (IEEE two-column format, 6 pages): [report/CMP684_Midterm_Karabay_N25220154.pdf](report/CMP684_Midterm_Karabay_N25220154.pdf)
- Code: [src/](src) (PyTorch; the method itself is in [src/tofe.py](src/tofe.py))
- Results behind every table: [runs/](runs)
- Trained models of all {len(published_runs(runs))} runs ({size:.0f} MB): [models/](models)

## What ToFe does

ToFe speeds up a vision transformer by using fewer tokens in the later blocks. Unlike pruning or
merging methods, it does not throw tokens away: a small MLP *freezes* unimportant tokens, another
small MLP updates them while they skip some blocks, and they can be *reused* later. A FLOPs loss
decides how many tokens each of the three stages keeps.

## Main results

DeiT-S on the full ImageNet-1k validation set (50,000 images), at about 50% of the FLOPs:

| Method | GMACs | Top-1 (%) |
|---|---:|---:|
| DeiT-S, no reduction | 4.60 | {b[('dense', '-')]['top1']:.2f} (paper: 79.9) |
| ToMe r=16, no training | {b[('ToMe', '16')]['gmacs']:.2f} | {b[('ToMe', '16')]['top1']:.2f} |
| EViT, no training | {b[('EViT', '[96, 49, 27]')]['gmacs']:.2f} | {b[('EViT', '[96, 49, 27]')]['top1']:.2f} |
| ToFe, my implementation (2 seeds) | {runs['tofe_b50']['gmacs']:.2f} | {mean(['tofe_b50', 'tofe_b50_s1']):.2f} (paper: 77.7) |
| ToFe without token reusing (2 seeds) | {runs['noreuse_b50']['gmacs']:.2f} | {mean(['noreuse_b50', 'noreuse_b50_s1']):.2f} |
| ToFe with fixed keep ratios (1/3 each) | {runs['fixed_33_33_33']['gmacs']:.2f} | {runs['fixed_33_33_33']['top1']:.2f} |

<img src="report/ieee/figs/tradeoff.svg" width="480" alt="Accuracy against cost">

Findings in short:

- The baselines match the paper (DeiT-S 79.85% vs. 79.9%, ToMe 76.6% vs. 76.5% at about 2.0 GMACs).
- Three of the four observations that motivate ToFe hold: early [CLS] attention is spread out,
  important tokens change between blocks, and neighbouring blocks are similar.
- The fourth one does not hold: tokens that look unimportant change by more than their own size
  over three blocks, so the small approximator cannot follow them and reused tokens are wrong.
- With about 59 times less training than the paper, ToFe stayed below the training-free baselines.
  Token reusing, the learned budget and the proposed selector were all worse than the alternatives.

## Repository layout

```
report/   the report (PDF), its source (ieee/paper.html) and figures
src/      all code (PyTorch); tofe.py contains the method
runs/     result files behind every table (JSON, training logs)
models/   trained checkpoints of all runs (see models/README.txt)
```

## How to run

Tested on one NVIDIA RTX 5070 (12 GB) with Python 3.13 and PyTorch 2.11 (CUDA 12.8).

```bash
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# ImageNet-1k from the Hugging Face hub (needs an account that accepted the dataset terms)
export HF_TOKEN=<your token>
bash src/download_imagenet.sh validation 0 13 14
bash src/download_imagenet.sh train 0 29 294
python src/prepare_imagenet.py && python src/resize_train.py && python src/cache_val.py

python src/test_tofe.py                      # unit tests
python src/motivation.py                     # motivating observations
python src/eval_baselines.py                 # dense model, EViT, ToMe, random tokens
python src/train_tofe.py --name tofe_b50     # one ToFe model (about 20 minutes)
bash src/run_queue.sh                        # all ablations
python src/analyze_tofe.py tofe_b50          # inference modes, token statistics
python src/diagnostics.py                    # why token reusing fails here
python src/paper_figures.py && python src/build_paper.py   # figures and the report PDF
```

To evaluate a trained model without training it again, only the validation data is needed:
`python src/analyze_tofe.py <run>` loads `models/<run>.pt` by itself. The report PDF is built
with headless Microsoft Edge.

## Notes

- The ImageNet data and the original paper PDF are not included (license and copyright).
- Every number in the report tables is read from the files in `runs/` (`src/paper_tables.py`).
"""


def build(runs, base, dst):
    os.makedirs(dst, exist_ok=True)
    for entry in os.listdir(dst):  # keep the git history, rebuild everything else
        if entry != '.git':
            path = os.path.join(dst, entry)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    os.makedirs(os.path.join(dst, 'src'))
    for f in glob.glob(os.path.join(ROOT, 'src', '*.py')) + glob.glob(os.path.join(ROOT, 'src', '*.sh')):
        shutil.copy(f, os.path.join(dst, 'src'))
    copy_results(runs, os.path.join(dst, 'runs'))
    os.makedirs(os.path.join(dst, 'models'))
    for n in published_runs(runs):
        shutil.copy(os.path.join(RUNS, n, 'model.pt'), os.path.join(dst, 'models', f'{n}.pt'))
    open(os.path.join(dst, 'models', 'README.txt'), 'w', encoding='utf-8').write(models_readme(runs))
    shutil.copytree(os.path.join(ROOT, 'report', 'ieee'), os.path.join(dst, 'report', 'ieee'),
                    ignore=shutil.ignore_patterns('_*'))
    shutil.copy(REPORT_PDF, os.path.join(dst, 'report'))
    open(os.path.join(dst, 'requirements.txt'), 'w').write(REQUIREMENTS)
    open(os.path.join(dst, '.gitignore'), 'w').write(GITIGNORE)
    open(os.path.join(dst, 'README.md'), 'w', encoding='utf-8').write(readme(runs, base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=os.path.join(ROOT, GITHUB_REPO.rstrip('/').split('/')[-1]))
    args = ap.parse_args()
    runs, base = load_runs(), json.load(open(os.path.join(RUNS, 'baselines.json')))
    build(runs, base, args.repo)
    files = [os.path.join(r, f) for r, ds, fs in os.walk(args.repo) if '.git' not in r.split(os.sep) for f in fs]
    biggest = max(files, key=os.path.getsize)
    print(f'{args.repo}: {len(files)} files, {sum(map(os.path.getsize, files)) / 2**20:.1f} MB, largest '
          f'{os.path.basename(biggest)} {os.path.getsize(biggest) / 2**20:.1f} MB')


if __name__ == '__main__':
    main()
