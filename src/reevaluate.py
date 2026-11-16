"""Re-evaluate finished runs after the selector-precision fix (fp32 logits, ranking by logit
difference). The previous numbers are kept in results.json as *_before_fix."""
import glob
import json
import os

import torch

import data
from analyze_tofe import load_run
from common import RUNS, checkpoint_path, evaluate, throughput


def main():
    for f in sorted(glob.glob(os.path.join(RUNS, '*', 'results.json'))):
        res = json.load(open(f))
        if 'top1_before_fix' in res:
            continue
        name = os.path.basename(os.path.dirname(f))
        model, keep = load_run(name)
        st = torch.load(checkpoint_path(name))['state']
        model.load_state_dict(st, strict=False)  # fine-tuned runs also store the backbone
        fn = lambda z: model.forward_infer(z, keep_counts=keep)
        top1, top5 = evaluate(fn, data.CachedVal(256))
        res.update(top1_before_fix=res['top1'], top5_before_fix=res['top5'], top1=top1, top5=top5,
                   throughput=throughput(fn))
        json.dump(res, open(f, 'w'), indent=1)
        print(f"{name:24s} {keep} {res['gmacs']:.2f} GMACs: {res['top1_before_fix']:.2f} -> {top1:.2f}", flush=True)


if __name__ == '__main__':
    main()
