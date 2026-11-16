"""Assemble the final PDF: report/report.html (+ syntax-highlighted source code appendix)
rendered with headless Edge, followed by the original paper.

usage: python src/build_report.py
"""
import glob
import os
import pathlib
import subprocess

import pypdf
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, PythonLexer

import report_tables
from common import ROOT

EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
REPORT = os.path.join(ROOT, 'report')
PAPER = next(iter(glob.glob(os.path.join(ROOT, 'paper', '*.pdf'))), None)  # not in the public repo
OUT = os.path.join(REPORT, 'ToFe_reproduction_report.pdf')
CODE_ORDER = ['tofe.py', 'baselines.py', 'train_tofe.py', 'analyze_tofe.py', 'diagnostics.py', 'motivation.py',
              'eval_baselines.py', 'test_tofe.py', 'feasibility.py', 'reevaluate.py', 'data.py', 'common.py',
              'plotstyle.py', 'make_figures.py', 'paper_figures.py', 'summarize.py', 'report_tables.py',
              'paper_tables.py', 'prepare_imagenet.py', 'resize_train.py', 'cache_val.py', 'download_imagenet.sh',
              'run_queue.sh', 'build_report.py', 'build_paper.py', 'make_submission.py', 'make_github_repo.py']


def code_appendix():
    fmt = HtmlFormatter(linenos='inline', style='friendly', cssclass='code')  # inline: numbers cannot drift at page breaks
    parts = [f'<style>{fmt.get_style_defs(".code")}</style>']
    for name in CODE_ORDER:
        path = os.path.join(ROOT, 'src', name)
        src = open(path, encoding='utf-8').read()
        lexer = BashLexer() if name.endswith('.sh') else PythonLexer()
        n = src.count('\n') + 1
        parts.append(f'<h3 class="codefile">src/{name} <span>({n} lines)</span></h3>{highlight(src, lexer, fmt)}')
    return '\n'.join(parts)


def main():
    html = open(os.path.join(REPORT, 'report.html'), encoding='utf-8').read()
    html = report_tables.fill(html).replace('<!--CODE_APPENDIX-->', code_appendix())
    assert '<!--TABLE:' not in html, 'unfilled table placeholder'
    full = os.path.join(REPORT, '_build.html')
    open(full, 'w', encoding='utf-8').write(html)
    body = os.path.join(REPORT, '_body.pdf')
    subprocess.run([EDGE, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    f'--print-to-pdf={body}', pathlib.Path(full).as_uri()], check=True, timeout=300)
    writer = pypdf.PdfWriter()
    report = pypdf.PdfReader(body)
    for p in report.pages:
        writer.add_page(p)
    writer.add_outline_item('Reproduction report', 0)
    start = len(writer.pages)
    for p in pypdf.PdfReader(PAPER).pages:
        writer.add_page(p)
    writer.add_outline_item('Original paper (Zhang et al., IEEE TNNLS 2026)', start)
    with open(OUT, 'wb') as f:
        writer.write(f)
    os.remove(body)
    print(f'{OUT}: {len(report.pages)} report pages + {len(writer.pages) - start} paper pages')


if __name__ == '__main__':
    main()
