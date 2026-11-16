"""Compact tables for the IEEE-format paper, generated from runs/*.json (see report_tables.py
for the helpers). Paper values are copied from Zhang et al. (TNNLS 2026)."""
from report_tables import ANA, DIAG, MOT, RUN, base, mean, ratios, seeds


def table(head, rows, aligns=None):
    aligns = aligns or ['l'] + ['r'] * (len(head) - 1)
    cell = lambda tag, c, a: f'<{tag} class="{a}">{c}</{tag}>'
    h = ''.join(cell('th', c, a) for c, a in zip(head, aligns))
    body = []
    for r in rows:
        if isinstance(r, str):
            body.append(f'<tr class="sec"><td colspan="{len(head)}">{r}</td></tr>')
        else:
            body.append('<tr>' + ''.join(cell('td', c, a) for c, a in zip(r, aligns)) + '</tr>')
    return f'<table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def t_choices():
    rows = [['Selector', 'LN, 384→96→96→2 MLP with ReLU; starts with p(keep) ≈ 0.88'],
            ['Gumbel-Softmax', 'on logits, τ = 1, hard + straight-through'],
            ['Approximator', 'LN, 384→192→384 MLP with GELU, last layer starts at zero'],
            ['L<sub>apr</sub>', 'mean squared error over frozen tokens'],
            ['L<sub>FLOPs</sub>', 'in GMACs, λ<sub>FLOPs</sub> = 50 (5 missed the budget)'],
            ['Training', '130,740 ImageNet images, 5 epochs, backbone frozen'],
            ['Optimizer', 'AdamW, lr 10<sup>−3</sup>, cosine, batch 128, bf16']]
    return table(['Item', 'Choice in this work'], rows, ['l', 'l'])


def t_motivation():
    m = MOT
    rows = [['Attention mass on top-50% tokens, blocks 2–5 / 10–12',
             f"{100 * min(m['top50_attention_mass'][1:5]):.0f}–{100 * max(m['top50_attention_mass'][1:5]):.0f}% / "
             f"{100 * min(m['top50_attention_mass'][9:]):.0f}–{100 * max(m['top50_attention_mass'][9:]):.0f}%", 'yes'],
            ['Top-25% tokens of block 10 removed by pruning at block 4',
             f"{100 * (1 - m['overlap']['recall@4 of top25%@10']):.0f}%", 'yes'],
            ['Cosine similarity of consecutive blocks',
             f"{min(m['cosine_similarity'][i][i + 1] for i in range(11)):.2f}–"
             f"{max(m['cosine_similarity'][i][i + 1] for i in range(11)):.2f}", 'yes'],
            ['Change over stage 1, inattentive / attentive tokens',
             f"{m['eps_per_stage']['inattentive'][0]:.2f} / {m['eps_per_stage']['attentive'][0]:.2f}", 'no'],
            ['cos(X<sub>3</sub>, X<sub>6</sub>) of inattentive tokens', f"{m['cos_per_stage']['inattentive'][0]:.2f}", 'no'],
            ['Norm growth ‖X<sub>6</sub>‖/‖X<sub>3</sub>‖ of inattentive tokens',
             f"{m['norm_ratio_per_stage']['inattentive'][0]:.2f}×", 'no']]
    return table(['Measurement', 'Value', 'Supports [1]?'], rows, ['l', 'r', 'c'])


def t_main():
    d, tofe = base('dense', '-'), ['tofe_b50', 'tofe_b50_s1']
    b = lambda m, s: base(m, s)
    rows = ['This work (RTX 5070)',
            ['DeiT-S (dense)', f"{d['gmacs']:.2f}", f"{d['top1']:.2f}", f"{d['throughput']:.0f}"],
            ['EViT, no training', f"{b('EViT', [96, 49, 27])['gmacs']:.2f}", f"{b('EViT', [96, 49, 27])['top1']:.2f}",
             f"{b('EViT', [96, 49, 27])['throughput']:.0f}"],
            ['ToMe r=16, no training', f"{b('ToMe', 16)['gmacs']:.2f}", f"{b('ToMe', 16)['top1']:.2f}",
             f"{b('ToMe', 16)['throughput']:.0f}"],
            ['ToMe r=18, no training', f"{b('ToMe', 18)['gmacs']:.2f}", f"{b('ToMe', 18)['top1']:.2f}",
             f"{b('ToMe', 18)['throughput']:.0f}"],
            ['Random tokens', f"{b('Random', [96, 49, 27])['gmacs']:.2f}", f"{b('Random', [96, 49, 27])['top1']:.2f}",
             f"{b('Random', [96, 49, 27])['throughput']:.0f}"],
            ['ToFe (2 seeds)', f"{mean(tofe, 'gmacs'):.2f}", seeds(tofe),
             f"{mean(tofe, 'throughput'):.0f}"],
            'Reported in [1] (RTX 4090)',
            ['DeiT-S (dense)', '4.6', '79.9', '2384'], ['EViT (trained)', '2.0', '68.6', '5442'],
            ['ToMe', '2.0', '76.5', '5610'], ['ToFe', '2.0', '77.7', '5477']]
    return table(['Method', 'GMACs', 'Top-1 (%)', 'img/s'], rows)


def t_ablation():
    p = lambda n: f"{RUN[n]['top1']:.2f}"
    g = lambda n: f"{RUN[n]['gmacs']:.2f}"
    tofe = ['tofe_b50', 'tofe_b50_s1']
    rows = [['ToFe default (reuse, MLP approx., learned budget)', f"{mean(tofe, 'gmacs'):.2f}", seeds(tofe), '78.48'],
            'Token reusing (Table V of [1], LV-ViT-S)',
            ['without reuse (2 seeds)', f"{mean(['noreuse_b50', 'noreuse_b50_s1'], 'gmacs'):.2f}",
             seeds(['noreuse_b50', 'noreuse_b50_s1']), '79.2'],
            'Approximator (Table VI of [1])',
            ['identity (plain reuse)', g('apx_identity'), p('apx_identity'), '77.09'],
            ['depth-wise conv', g('apx_dwconv'), p('apx_dwconv'), '77.33'],
            ['transformer block', g('apx_block'), p('apx_block'), '74.27'],
            'Fixed keep ratios instead of learned budget (Table VII of [1])',
            ['(0.5, 0.25, 0.125)', g('fixed_50_25_12'), p('fixed_50_25_12'), '78.35'],
            ['(0.4, 0.4, 0.1)', g('fixed_40_40_10'), p('fixed_40_40_10'), '78.21'],
            ['(1/3, 1/3, 1/3)', g('fixed_33_33_33'), p('fixed_33_33_33'), '76.78'],
            ['(0.4, 0.3, 0.3)', g('fixed_40_30_30'), p('fixed_40_30_30'), '77.86'],
            'Selector (Table IV of [1])',
            ['5-layer MLP', g('sel_depth5'), p('sel_depth5'), '78.51'],
            ['7-layer MLP', g('sel_depth7'), p('sel_depth7'), '78.42'],
            ['DynamicViT predictor', g('sel_dynamicvit'), p('sel_dynamicvit'), '74.35']]
    return table(['Variant', 'GMACs', 'Ours', '[1]'], rows)


def t_budget():
    rows = []
    for n in ['tofe_b40', 'tofe_b50', 'tofe_b60', 'tofe_b70']:
        r = RUN[n]
        rows.append([f"{int(n[-2:])}%", f"{r['gmacs']:.2f}", ', '.join(map(str, r['keep_counts'])), f"{r['top1']:.2f}"])
    return table(['Budget', 'GMACs', 'Kept tokens / stage', 'Top-1'], rows)


def t_training():
    rows = [['5 epochs (default)', '0.65 M', seeds(['tofe_b50', 'tofe_b50_s1'])],
            ['15 epochs', '1.96 M', f"{RUN['tofe_b50_e15']['top1']:.2f}"],
            ['5 + 3 epochs backbone fine-tuning', '1.05 M', f"{RUN['tofe_b50_ft3']['top1']:.2f}"],
            ['[1]: 30 epochs, full ImageNet', '38.4 M', '77.7–78.5']]
    return table(['Training', 'Samples', 'Top-1'], rows)


def t_reuse_probe():
    rows = []
    for name, label in [('tofe_b50', '5 epochs'), ('tofe_b50_e15', '15 epochs'), ('tofe_b50_ft3', '+ fine-tune')]:
        r = DIAG[name]['D1']['paper (0.49,0.25,0.14)']
        rows.append([label, f"{r['reuse']:.2f}", f"{r['no_reuse']:.2f}"])
    return table(['Model', 'Reuse on', 'Reuse off'], rows)


def t_inside():
    st, d = ANA['stages'], DIAG['tofe_b50']
    rows = [['Kept tokens in stage 1 / 2 / 3', ' / '.join(f"{s['remaining_mean']:.0f}" for s in st), 'n/a / 49 / 26'],
            ['Stage-2 kept tokens that were frozen before', f"{100 * st[1]['reused_ratio']:.1f}%", '10.9%'],
            ['AEP in stage 2 / 3', f"{100 * st[1]['AEP']:.0f}% / {100 * st[2]['AEP']:.0f}%", '91% / 87%'],
            ['Error of frozen tokens, identity → approx.',
             f"{d['D2']['stage1']['identity']:.2f} → {d['D2']['stage1']['approximator']:.2f}", 'n/a'],
            ['Pruning top-1: selector score / [CLS] attention',
             f"{d['D3']['prune_top1']['selector']:.1f} / {d['D3']['prune_top1']['cls_attention']:.1f}", 'n/a']]
    return table(['Quantity (5-epoch model)', 'Ours', '[1]'], rows)


def t_speed():
    d, t = base('dense', '-'), RUN['tofe_b50']
    bs, lat = ANA['batch_size'], ANA['latency_ms_bs1']
    rows = [['Top-1, batch-adaptive, batch 16 / 128', f"{bs['16']['top1']:.2f} / {bs['128']['top1']:.2f}",
             '78.58 / 78.48'],
            ['Top-1, instance-adaptive (batch 1)', f"{bs['1']['top1']:.2f}", '78.62'],
            ['Throughput gain, batch 128', f"+{100 * (t['throughput'] / d['throughput'] - 1):.0f}%", '+130%'],
            ['Latency, batch 1 (dense → ToFe)', f'{lat[0]:.1f} → {lat[1]:.1f} ms', 'n/a']]
    return table(['Inference', 'Ours', '[1]'], rows)


TABLES = {name[2:]: fn for name, fn in globals().items() if name.startswith('t_')}


def fill(html):
    for name, fn in TABLES.items():
        html = html.replace(f'<!--TABLE:{name}-->', fn())
    return html
