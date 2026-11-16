"""Collect every finished run and baseline into runs/summary.md (the numbers used in the report)."""
import json
import os

from common import RUNS
from make_figures import load_runs


def main():
    runs = load_runs()
    base = json.load(open(os.path.join(RUNS, 'baselines.json')))
    lines = ['| run | keep N^s | GMACs | top-1 | top-5 | img/s |', '|---|---|---|---|---|---|']
    for b in base:
        lines.append(f"| {b['method']} {b['setting']} | - | {b['gmacs']:.2f} | {b['top1']:.2f} | {b['top5']:.2f} | "
                     f"{b['throughput']:.0f} |")
    for name in sorted(runs):
        r = runs[name]
        lines.append(f"| {name} | {r['keep_counts']} | {r['gmacs']:.2f} | {r['top1']:.2f} | {r['top5']:.2f} | "
                     f"{r['throughput']:.0f} |")
    text = '\n'.join(lines)
    open(os.path.join(RUNS, 'summary.md'), 'w', encoding='utf-8').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
