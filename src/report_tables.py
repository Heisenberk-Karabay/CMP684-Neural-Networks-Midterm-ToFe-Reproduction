"""HTML tables for the report, generated from the result files so that every number in the
report comes from runs/*.json. Paper values are copied from Zhang et al. (TNNLS 2026)."""
import json
import os

from common import RUNS
from make_figures import load_runs

RUN = load_runs()
BASE = json.load(open(os.path.join(RUNS, 'baselines.json')))
MOT = json.load(open(os.path.join(RUNS, 'motivation.json')))
DIAG = json.load(open(os.path.join(RUNS, 'diagnostics.json')))
ANA = json.load(open(os.path.join(RUNS, 'tofe_b50', 'analysis.json')))


def base(method, setting):
    return next(b for b in BASE if b['method'] == method and b['setting'] == setting)


def mean(names, key='top1'):
    vals = [RUN[n][key] for n in names]
    return sum(vals) / len(vals)


def seeds(names):
    vals = [RUN[n]['top1'] for n in names]
    m = sum(vals) / len(vals)
    return f'{m:.2f} ± {max(abs(v - m) for v in vals):.2f}' if len(vals) > 1 else f'{m:.2f}'


def table(head, rows, cls=''):
    h = ''.join(f'<th>{c}</th>' for c in head)
    body = []
    for r in rows:
        if isinstance(r, str):  # section row
            body.append(f'<tr class="sec"><td colspan="{len(head)}">{r}</td></tr>')
        else:
            body.append('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>')
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def ratios(keep, n=196):
    return '(' + ', '.join(f'{k / n:.2f}' for k in keep) + ')'


def t_main():
    d = base('dense', '-')
    tofe = ['tofe_b50', 'tofe_b50_s1']
    rows = [
        'Ours: DeiT-S on the full ImageNet-1k val set (50,000 images), RTX 5070, bf16, batch 128',
        ['DeiT-S (dense)', '4.60', f"{d['top1']:.2f}", f"{d['top5']:.2f}", f"{d['throughput']:.0f}", 'n/a'],
        ['EViT, training-free, keep (0.49, 0.25, 0.14)', f"{base('EViT', [96, 49, 27])['gmacs']:.2f}",
         f"{base('EViT', [96, 49, 27])['top1']:.2f}", f"{base('EViT', [96, 49, 27])['top5']:.2f}",
         f"{base('EViT', [96, 49, 27])['throughput']:.0f}", 'n/a'],
        ['ToMe, training-free, r = 16', f"{base('ToMe', 16)['gmacs']:.2f}", f"{base('ToMe', 16)['top1']:.2f}",
         f"{base('ToMe', 16)['top5']:.2f}", f"{base('ToMe', 16)['throughput']:.0f}", 'n/a'],
        ['ToMe, training-free, r = 18', f"{base('ToMe', 18)['gmacs']:.2f}", f"{base('ToMe', 18)['top1']:.2f}",
         f"{base('ToMe', 18)['top5']:.2f}", f"{base('ToMe', 18)['throughput']:.0f}", 'n/a'],
        ['Random token pruning (sanity check)', f"{base('Random', [96, 49, 27])['gmacs']:.2f}",
         f"{base('Random', [96, 49, 27])['top1']:.2f}", f"{base('Random', [96, 49, 27])['top5']:.2f}",
         f"{base('Random', [96, 49, 27])['throughput']:.0f}", 'n/a'],
        [f'<b>ToFe (ours), learned budget, 2 seeds</b>', f"{mean(tofe, 'gmacs'):.2f}", f'<b>{seeds(tofe)}</b>',
         f"{mean(tofe, 'top5'):.2f}", f"{mean(tofe, 'throughput'):.0f}", f"+{RUN['tofe_b50']['extra_params_M']:.2f}M"],
        ['ToFe (ours), fixed ratios (1/3, 1/3, 1/3)', f"{RUN['fixed_33_33_33']['gmacs']:.2f}",
         f"{RUN['fixed_33_33_33']['top1']:.2f}", f"{RUN['fixed_33_33_33']['top5']:.2f}",
         f"{RUN['fixed_33_33_33']['throughput']:.0f}", f"+{RUN['fixed_33_33_33']['extra_params_M']:.2f}M"],
        'Paper, Table I (DeiT-S, RTX 4090, batch 128; trained baselines)',
        ['DeiT-S (dense)', '4.6', '79.9', '95.0', '2383.9', 'n/a'],
        ['EViT', '2.0', '68.6', '86.2', '5442.1', 'n/a'],
        ['DynamicViT', '2.1', '75.2', '92.1', '5607.9', '+0.7M'],
        ['ToMe', '2.0', '76.5', '92.8', '5610.1', 'n/a'],
        ['IdleViT', '2.0', '77.6', '93.8', '5603.7', 'n/a'],
        ['<b>ToFe</b>', '2.0', '<b>77.7</b>', '<b>94.1</b>', '5477.3', '+0.2M'],
    ]
    return table(['Method', 'GFLOPs / GMACs', 'Top-1 (%)', 'Top-5 (%)', 'img/s', 'Extra params'], rows)


def t_motivation():
    m = MOT
    rows = [
        ['M1', '[CLS] attention mass on the top-50% tokens, blocks 2–5 vs. 10–12',
         f"{100 * min(m['top50_attention_mass'][1:5]):.0f}–{100 * max(m['top50_attention_mass'][1:5]):.0f}% vs. "
         f"{100 * min(m['top50_attention_mass'][9:]):.0f}–{100 * max(m['top50_attention_mass'][9:]):.0f}%",
         'Fig. 3', '✔'],
        ['M2', 'Share of block-10 top-25% tokens that top-50% pruning at block 4 would remove',
         f"{100 * (1 - m['overlap']['recall@4 of top25%@10']):.0f}% (they carry "
         f"{100 * m['overlap']['block-10 attn mass outside top50%@4']:.0f}% of block-10 attention)", 'Fig. 3, 4a', '✔'],
        ['M3', 'Mean token cosine similarity of consecutive block outputs',
         f"{min(m['cosine_similarity'][i][i + 1] for i in range(11)):.2f}–"
         f"{max(m['cosine_similarity'][i][i + 1] for i in range(11)):.2f}", 'Fig. 4b', '✔'],
        ['M4', 'Relative change ‖X<sub>l+3</sub> − X<sub>l</sub>‖/‖X<sub>l</sub>‖ over stage 1 / 2 '
               '(inattentive | attentive tokens)',
         f"{m['eps_per_stage']['inattentive'][0]:.2f} / {m['eps_per_stage']['inattentive'][1]:.2f} | "
         f"{m['eps_per_stage']['attentive'][0]:.2f} / {m['eps_per_stage']['attentive'][1]:.2f}", 'Sec. III-B, IV-B', '✘'],
        ['M4', 'cos(X<sub>l</sub>, X<sub>l+3</sub>) over stage 1 / 2 (inattentive tokens)',
         f"{m['cos_per_stage']['inattentive'][0]:.2f} / {m['cos_per_stage']['inattentive'][1]:.2f}", '', '✘'],
        ['M4', '‖X<sub>l+3</sub>‖/‖X<sub>l</sub>‖ over stage 1 / 2 (inattentive tokens)',
         f"{m['norm_ratio_per_stage']['inattentive'][0]:.2f} / {m['norm_ratio_per_stage']['inattentive'][1]:.2f}", '',
         '✘'],
    ]
    return table(['', 'Measured quantity (DeiT-S, 5,000 val images)', 'Ours', 'Paper', 'Supports claim'], rows)


def t_reuse():
    rows = [
        'Ours (DeiT-S, 50% budget, 5 epochs, 2 seeds)',
        ['ToFe w/o token reusing', f"{mean(['noreuse_b50', 'noreuse_b50_s1'], 'gmacs'):.2f}",
         seeds(['noreuse_b50', 'noreuse_b50_s1']), str(RUN['noreuse_b50']['keep_counts'])],
        ['ToFe w/ token reusing', f"{mean(['tofe_b50', 'tofe_b50_s1'], 'gmacs'):.2f}",
         seeds(['tofe_b50', 'tofe_b50_s1']), str(RUN['tofe_b50']['keep_counts'])],
        'Paper, Table V (LV-ViT-S, dense 83.3)',
        ['ToFe w/o token reusing', '3.3', '79.2', 'n/a'],
        ['ToFe w/ token reusing', '3.3', '81.9', 'n/a'],
    ]
    return table(['Model', 'GMACs', 'Top-1 (%)', 'Learned N<sup>s</sup>'], rows)


def t_reuse_probe():
    rows = []
    for name, label in [('tofe_b50', '5 epochs'), ('tofe_b50_e15', '15 epochs'), ('tofe_b50_ft3', '+ backbone ft')]:
        for alloc, r in DIAG[name]['D1'].items():
            nr = f"{r['no_reuse']:.2f}" if 'no_reuse' in r else 'n/a (not nested)'
            rows.append([label, f"{alloc} {r['keep']}", f"{r['gmacs']:.2f}", f"{r['reuse']:.2f}", nr])
    return table(['Trained model', 'Kept tokens N<sup>s</sup> at inference', 'GMACs', 'Reuse on', 'Reuse off'], rows)


def t_diag():
    rows = []
    for name, label in [('tofe_b50', '5 epochs'), ('tofe_b50_e15', '15 epochs'), ('tofe_b50_ft3', '+ backbone ft')]:
        d = DIAG[name]
        q = d['D3']['keep_prob_quantiles_10_25_50_75_90']
        rows.append([label, f"{d['D2']['stage1']['identity']:.2f} → {d['D2']['stage1']['approximator']:.2f}",
                     f"{d['D2']['stage2']['identity']:.2f} → {d['D2']['stage2']['approximator']:.2f}",
                     f"{q[1]:.2f} / {q[2]:.2f} / {q[3]:.2f}",
                     f"{d['D3']['prune_top1']['selector']:.2f}", f"{d['D3']['prune_top1']['cls_attention']:.2f}"])
    return table(['Model', 'Rel. error, stage 1<br>identity → approx.', 'Rel. error, stage 2<br>identity → approx.',
                  'Keep prob. quartiles<br>(stage 1)', 'Prune by<br>selector', 'Prune by<br>[CLS] attn.'], rows)


def t_approx():
    rows = [
        'Ours (DeiT-S, 50% budget, 5 epochs)',
        ['Bottleneck MLP (D → D/2 → D)', f"{RUN['tofe_b50']['gmacs']:.2f}", ratios(RUN['tofe_b50']['keep_counts']),
         seeds(['tofe_b50', 'tofe_b50_s1'])],
        ['Identity (direct reuse)', f"{RUN['apx_identity']['gmacs']:.2f}", ratios(RUN['apx_identity']['keep_counts']),
         f"{RUN['apx_identity']['top1']:.2f}"],
        ['Depth-wise 3×3 conv', f"{RUN['apx_dwconv']['gmacs']:.2f}", ratios(RUN['apx_dwconv']['keep_counts']),
         f"{RUN['apx_dwconv']['top1']:.2f}"],
        ['Transformer block', f"{RUN['apx_block']['gmacs']:.2f}", ratios(RUN['apx_block']['keep_counts']),
         f"{RUN['apx_block']['top1']:.2f}"],
        'Paper, Table VI (DeiT-S)',
        ['Bottleneck MLP', '2.3', '(0.49, 0.25, 0.14)', '78.48'],
        ['Identity', '2.3', '(0.59, 0.26, 0.14)', '77.09'],
        ['Depth-wise conv', '2.3', '(0.55, 0.25, 0.14)', '77.33'],
        ['Transformer block', '2.4', '(0.27, 0.11, 0.07)', '74.27'],
    ]
    return table(['Approximator', 'GMACs', 'Keep ratios', 'Top-1 (%)'], rows)


def t_selector():
    rows = [
        'Ours (DeiT-S, 50% budget, 5 epochs)',
        ['3-layer MLP (ToFe)', '1', f"{RUN['tofe_b50']['extra_params_M']:.2f}", ratios(RUN['tofe_b50']['keep_counts']),
         seeds(['tofe_b50', 'tofe_b50_s1'])],
        ['5-layer MLP', '3', f"{RUN['sel_depth5']['extra_params_M']:.2f}", ratios(RUN['sel_depth5']['keep_counts']),
         f"{RUN['sel_depth5']['top1']:.2f}"],
        ['7-layer MLP', '5', f"{RUN['sel_depth7']['extra_params_M']:.2f}", ratios(RUN['sel_depth7']['keep_counts']),
         f"{RUN['sel_depth7']['top1']:.2f}"],
        ['DynamicViT predictor', '2', f"{RUN['sel_dynamicvit']['extra_params_M']:.2f}",
         ratios(RUN['sel_dynamicvit']['keep_counts']), f"{RUN['sel_dynamicvit']['top1']:.2f}"],
        'Paper, Table IV (DeiT-S, 2.3–2.5 GFLOPs)',
        ['3-layer MLP (ToFe)', '1', '22.1M total', '(0.49, 0.25, 0.14)', '78.48'],
        ['5-layer MLP', '3', '22.3M total', '(0.49, 0.25, 0.14)', '78.51'],
        ['7-layer MLP', '5', '22.4M total', '(0.51, 0.26, 0.13)', '78.42'],
        ['DynamicViT', '2', '22.8M total', '(0.47, 0.21, 0.11)', '74.35'],
    ]
    return table(['Selector', 'Hidden layers', 'Extra params (M)', 'Keep ratios', 'Top-1 (%)'], rows)


def t_fixed():
    paper = {'tofe_b50': '78.48', 'fixed_50_25_12': '78.35', 'fixed_40_40_10': '78.21', 'fixed_33_33_33': '76.78',
             'fixed_40_30_30': '77.86'}
    rows = [[f"Optimized (learned): {ratios(RUN['tofe_b50']['keep_counts'])}", f"{RUN['tofe_b50']['gmacs']:.2f}",
             seeds(['tofe_b50', 'tofe_b50_s1']), '(0.49, 0.25, 0.14): ' + paper['tofe_b50']]]
    for n, lab in [('fixed_50_25_12', '(0.5, 0.25, 0.125)'), ('fixed_40_40_10', '(0.4, 0.4, 0.1)'),
                   ('fixed_33_33_33', '(0.33, 0.33, 0.33)'), ('fixed_40_30_30', '(0.4, 0.3, 0.3)')]:
        rows.append([f'Fixed {lab}', f"{RUN[n]['gmacs']:.2f}", f"{RUN[n]['top1']:.2f}", paper[n]])
    return table(['Keep ratio', 'GMACs (ours)', 'Top-1 ours (%)', 'Top-1 paper, Table VII (%)'], rows)


def t_budget():
    rows = []
    for n in ['tofe_b40', 'tofe_b50', 'tofe_b60', 'tofe_b70']:
        r = RUN[n]
        rows.append([f"{int(n[-2:])}%", f"{r['gmacs']:.2f}", str(r['keep_counts']), f"{r['top1']:.2f}",
                     f"{r['throughput']:.0f}"])
    return table(['Budget', 'GMACs', 'Learned N<sup>s</sup>', 'Top-1 (%)', 'img/s'], rows)


def t_training():
    rows = [['5 epochs, frozen backbone (2 seeds)', '0.65M', f"{mean(['tofe_b50', 'tofe_b50_s1'], 'gmacs'):.2f}",
             str(RUN['tofe_b50']['keep_counts']), seeds(['tofe_b50', 'tofe_b50_s1'])],
            ['15 epochs, frozen backbone', '1.96M', f"{RUN['tofe_b50_e15']['gmacs']:.2f}",
             str(RUN['tofe_b50_e15']['keep_counts']), f"{RUN['tofe_b50_e15']['top1']:.2f}"],
            ['5 frozen + 3 epochs backbone fine-tuning', '1.05M', f"{RUN['tofe_b50_ft3']['gmacs']:.2f}",
             str(RUN['tofe_b50_ft3']['keep_counts']), f"{RUN['tofe_b50_ft3']['top1']:.2f}"],
            ['Paper: 5 frozen + 25 epochs fine-tuning, full ImageNet', '38.4M', '2.0–2.3', '(96, 49, 27)',
             '77.7–78.48']]
    return table(['Training schedule', 'Samples seen', 'GMACs', 'Learned N<sup>s</sup>', 'Top-1 (%)'], rows)


def t_batch():
    bs = ANA['batch_size']
    paper = {'1': '78.62', '16': '78.58', '32': '78.51', '64': '78.45', '128': '78.48'}
    rows = [[k if k != '1' else '1 (instance-adaptive)', f"{bs[k]['top1']:.2f}", f"{bs[k]['gmacs']:.2f}", paper[k]]
            for k in ['1', '16', '32', '64', '128']]
    return table(['Batch size', 'Top-1 ours (%)', 'GMACs ours', 'Top-1 paper, Table VIII (%)'], rows)


def t_reuse_stats():
    rows = []
    for s, st in enumerate(ANA['stages']):
        rows.append([f'Stage {s + 1}', f"{st['remaining_mean']:.1f} ± {st['remaining_std']:.1f}",
                     f"{100 * st['AEP']:.1f}%", f"{st['reused_mean']:.2f}", f"{100 * st['reused_ratio']:.1f}%"])
    rows += ['Paper, Table X (DeiT-S)', ['Stage 2', '48.71', '91.4%', '5.32', '10.92%'],
             ['Stage 3', '26.43', '87.2%', '2.68', '10.14%']]
    return table(['', '# remaining tokens', 'AEP', '# reused tokens', 'Reused ratio'], rows)


def t_efficiency():
    d, t = base('dense', '-'), RUN['tofe_b50']
    lat = ANA['latency_ms_bs1']
    rows = [['Throughput, batch 128 (img/s)', f"{d['throughput']:.0f}", f"{t['throughput']:.0f}",
             f"{100 * (t['throughput'] / d['throughput'] - 1):+.0f}%", '2383.9 → 5477.3 (+130%)*'],
            ['Latency, batch 1, instance-adaptive (ms)', f'{lat[0]:.2f}', f'{lat[1]:.2f}',
             f'{100 * (lat[1] / lat[0] - 1):+.0f}%', 'not reported']]
    return table(['', 'Dense DeiT-S', 'ToFe (50%)', 'Change', 'Paper (DeiT-S, RTX 4090)'], rows)


TABLES = {name[2:]: fn for name, fn in globals().items() if name.startswith('t_')}


def fill(html):
    for name, fn in TABLES.items():
        html = html.replace(f'<!--TABLE:{name}-->', fn())
    return html


if __name__ == '__main__':
    for name, fn in TABLES.items():
        fn()
    print('tables:', ', '.join(TABLES))
