"""Assemble the files to submit to the instructor (usage: python src/make_submission.py, after build_paper.py):

  submission/CMP684_Midterm_Karabay_N25220154/      report, original paper, code, results, figures, README
  submission/CMP684_Midterm_Karabay_N25220154.zip   the same folder as one archive

The trained models are published in the project repository (common.PROJECT_URL, see make_github_repo.py).
"""
import glob
import json
import os
import shutil

from build_report import PAPER
from build_paper import CODE_PDF, REPORT_PDF
from common import PROJECT_URL, ROOT, RUNS
from make_figures import load_runs

NAME = 'CMP684_Midterm_Karabay_N25220154'
OUT = os.path.join(ROOT, 'submission')
PKG = os.path.join(OUT, NAME)
SKIP_RUNS = {'old_fixed_topk_50_25_12'}  # first fixed-ratio method, replaced (see the report, Sec. VI)
REQUIREMENTS = """# Python 3.13. Install PyTorch with CUDA 12.8 first:
#   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
timm==1.0.30
numpy==2.5.2
pillow==12.3.0
matplotlib==3.11.2
pyarrow==26.0.0
fvcore==0.1.5.post20221221
pypdf==6.21.2
pygments==2.21.0
pypdfium2==5.14.0
"""


def published_runs(runs):
    return [n for n in sorted(runs) if n not in SKIP_RUNS]


def readme(runs, base):
    b = {(x['method'], str(x['setting'])): x for x in base}
    mean = lambda names: sum(runs[n]['top1'] for n in names) / len(names)
    size = sum(os.path.getsize(os.path.join(RUNS, n, 'model.pt')) for n in published_runs(runs)) / 2**20
    return f"""# CMP684 Neural Networks - Midterm Reading Project

Hasan Berk Karabay (N25220154)

Paper studied: H. Zhang, J. Zhang, C. Hu, S. Guo, "ToFe: Lagged Token Freezing and Reusing
for Efficient Vision Transformer Inference", IEEE Transactions on Neural Networks and Learning
Systems, vol. 37, no. 10, pp. 5545-5558, Oct. 2026. The authors did not release code; all code
here was written for this project.

Online repository (code, results and all {len(published_runs(runs))} trained models, {size:.0f} MB):
{PROJECT_URL}


## Contents

  1_Report_Karabay_N25220154.pdf     my report (IEEE two-column format, 6 pages)
  2_Original_Paper_Zhang2026_ToFe.pdf the paper that was reproduced
  3_Source_Code/                      all source code (Python / PyTorch) + requirements.txt
  3_Source_Code_Listing.pdf           the same code as a printable PDF
  4_Results/                          the result files behind every table (JSON + training logs)
  5_Figures/                          the figures of the report


## Main results (DeiT-S, full ImageNet-1k validation set, about 50% of the FLOPs)

  DeiT-S (no reduction)            4.60 GMACs   {b[('dense', '-')]['top1']:.2f}%   (paper: 79.9%)
  ToMe r=16, no training           {b[('ToMe', '16')]['gmacs']:.2f} GMACs   {b[('ToMe', '16')]['top1']:.2f}%
  EViT, no training                {b[('EViT', '[96, 49, 27]')]['gmacs']:.2f} GMACs   {b[('EViT', '[96, 49, 27]')]['top1']:.2f}%
  ToFe, my implementation          {runs['tofe_b50']['gmacs']:.2f} GMACs   {mean(['tofe_b50', 'tofe_b50_s1']):.2f}%   (paper: 77.7-78.5%)
  ToFe without token reusing       {runs['noreuse_b50']['gmacs']:.2f} GMACs   {mean(['noreuse_b50', 'noreuse_b50_s1']):.2f}%
  ToFe, fixed ratios (1/3 each)    {runs['fixed_33_33_33']['gmacs']:.2f} GMACs   {runs['fixed_33_33_33']['top1']:.2f}%

With about 59x less training than the paper, ToFe did not reach the reported accuracy, and
token reusing, the learned budget and the proposed selector were all worse than the
alternatives. The report explains why (Sec. V-F).


## How to run

Hardware used: one NVIDIA RTX 5070 (12 GB), Windows 11, Python 3.13.

1. Install: see 3_Source_Code/requirements.txt.
2. Data: ImageNet-1k from the Hugging Face hub (ILSVRC/imagenet-1k, needs an account that
   accepted the dataset terms; the data itself is not included here):
     export HF_TOKEN=<your token>
     bash src/download_imagenet.sh validation 0 13 14
     bash src/download_imagenet.sh train 0 29 294
     python src/prepare_imagenet.py
     python src/resize_train.py
     python src/cache_val.py
3. Tests:          python src/test_tofe.py
4. Motivation:     python src/motivation.py                (Table II, Figs. 1-4)
5. Baselines:      python src/eval_baselines.py            (dense, EViT, ToMe, random)
6. ToFe:           python src/train_tofe.py --name tofe_b50 (about 20 min)
                   bash src/run_queue.sh                    (all ablations)
7. Analysis:       python src/analyze_tofe.py tofe_b50
                   python src/diagnostics.py
8. Report:         python src/paper_figures.py
                   python src/build_paper.py

Run the commands from the folder that contains src/. Results go to runs/<name>/, and every
number in the report tables is read from these files (src/paper_tables.py).

To use a trained model, put the models/ folder of the online repository next to src/.
load_run(<name>) in src/analyze_tofe.py then finds models/<name>.pt by itself, and
"python src/analyze_tofe.py <name>" evaluates it on the validation set.


## Source files

  tofe.py              the ToFe method: selector, approximator, training and inference passes
  baselines.py         EViT and ToMe (training-free)
  train_tofe.py        training and evaluation of one ToFe model
  analyze_tofe.py      batch-size / inference-mode study, token statistics, Fig. 7
  diagnostics.py       why reuse fails: reuse on/off, approximation error, selector quality
  motivation.py        the motivating observations of the paper
  eval_baselines.py    dense model and baselines on the validation set
  test_tofe.py         unit tests
  feasibility.py       first feasibility test on Imagenette / Imagewoof
  reevaluate.py        re-scoring of all runs after the bug fix in Sec. VI
  data.py, common.py   data loading and shared helpers
  *_figures.py, *_tables.py, plotstyle.py, summarize.py, build_*.py   figures, tables, PDFs
  prepare_imagenet.py, resize_train.py, cache_val.py, download_imagenet.sh   data preparation
  run_queue.sh         list of all ablation runs
  make_submission.py, make_github_repo.py   build this package and the GitHub repository
"""


def copy_results(runs, dst):
    os.makedirs(dst)
    for f in ['summary.md', 'baselines.json', 'motivation.json', 'diagnostics.json', 'feasibility_ce.log',
              'feasibility_kd.log']:
        shutil.copy(os.path.join(RUNS, f), dst)
    for n in runs:
        os.makedirs(os.path.join(dst, n))
        for f in ['results.json', 'log.txt', 'analysis.json']:
            if os.path.exists(os.path.join(RUNS, n, f)):
                shutil.copy(os.path.join(RUNS, n, f), os.path.join(dst, n))


def main():
    runs, base = load_runs(), json.load(open(os.path.join(RUNS, 'baselines.json')))
    for old in (PKG, os.path.join(OUT, 'trained_models')):  # trained_models: earlier Drive package
        if os.path.isdir(old):
            shutil.rmtree(old)
    if os.path.exists(os.path.join(OUT, 'trained_models.zip')):
        os.remove(os.path.join(OUT, 'trained_models.zip'))

    os.makedirs(PKG)
    shutil.copy(REPORT_PDF, os.path.join(PKG, '1_Report_Karabay_N25220154.pdf'))
    shutil.copy(PAPER, os.path.join(PKG, '2_Original_Paper_Zhang2026_ToFe.pdf'))
    code = os.path.join(PKG, '3_Source_Code', 'src')
    os.makedirs(code)
    for f in glob.glob(os.path.join(ROOT, 'src', '*.py')) + glob.glob(os.path.join(ROOT, 'src', '*.sh')):
        shutil.copy(f, code)
    open(os.path.join(PKG, '3_Source_Code', 'requirements.txt'), 'w').write(REQUIREMENTS)
    shutil.copy(CODE_PDF, os.path.join(PKG, '3_Source_Code_Listing.pdf'))
    copy_results(runs, os.path.join(PKG, '4_Results'))
    shutil.copytree(os.path.join(ROOT, 'report', 'ieee', 'figs'), os.path.join(PKG, '5_Figures'))
    open(os.path.join(PKG, 'README.md'), 'w', encoding='utf-8').write(readme(runs, base))
    shutil.make_archive(PKG, 'zip', OUT, NAME)
    print(f'{PKG}.zip: {os.path.getsize(PKG + ".zip") / 2**20:.1f} MB')


if __name__ == '__main__':
    main()
