Trained ToFe checkpoints, one file per run (DeiT-S backbone from timm,
deit_small_patch16_224.fb_in1k). Each file is a dict with "args" (training options),
"keep_avg" (average kept tokens per stage) and "state" (weights of the token selectors and
approximators; tofe_b50_ft3 also contains its fine-tuned backbone).

The code finds these files by itself (src/common.py, checkpoint_path):
  from analyze_tofe import load_run; model, keep = load_run("tofe_b50")
  python src/analyze_tofe.py tofe_b50      # evaluates it on the ImageNet validation set

run                     GMACs   top-1  kept tokens per stage
apx_block                2.30   47.49  [0, 0, 57]
apx_dwconv               2.32   70.32  [120, 61, 15]
apx_identity             2.32   69.07  [112, 69, 16]
fixed_33_33_33           2.34   72.68  [65, 65, 65]
fixed_40_30_30           2.35   72.65  [78, 59, 59]
fixed_40_40_10           2.24   70.44  [78, 78, 20]
fixed_50_25_12           2.21   72.08  [98, 49, 24]
noreuse_b50              2.32   71.84  [140, 31, 17]
noreuse_b50_s1           2.32   71.63  [140, 31, 17]
sel_depth5               2.32   70.35  [99, 76, 14]
sel_depth7               2.32   70.28  [101, 72, 14]
sel_dynamicvit           2.32   71.55  [104, 52, 14]
tofe_b40                 1.87   69.95  [65, 31, 13]
tofe_b50                 2.32   69.63  [98, 79, 14]
tofe_b50_e15             2.32   70.74  [90, 87, 14]
tofe_b50_ft3             2.33   71.23  [67, 114, 11]
tofe_b50_s1              2.33   69.69  [100, 78, 14]
tofe_b60                 2.77   74.79  [47, 196, 23]
tofe_b70                 3.24   76.24  [196, 135, 19]
