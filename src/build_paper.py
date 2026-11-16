"""Build the IEEE two-column report (report/ieee/paper.html, tables filled from the result files)
and, as a separate PDF, a printable listing of the source code. Both are rendered with headless Edge.

usage: python src/build_paper.py [--code-url URL]  (default: common.PROJECT_URL)   (run src/paper_figures.py first)
"""
import argparse
import os
import pathlib
import re
import subprocess

import pypdf

import paper_tables
from build_report import EDGE, code_appendix
from common import PROJECT_URL, ROOT

IEEE = os.path.join(ROOT, 'report', 'ieee')
REPORT_PDF = os.path.join(ROOT, 'report', 'CMP684_Midterm_Karabay_N25220154.pdf')
CODE_PDF = os.path.join(ROOT, 'report', 'source_code_listing.pdf')

CODE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 15mm 14mm 16mm 14mm;
  @bottom-center {{ content: "Source code, page " counter(page); font: 8pt "Times New Roman", serif; }} }}
body {{ font: 9pt "Times New Roman", serif; margin: 0; }}
h1 {{ font-size: 13pt; margin: 0 0 3pt; }} p {{ margin: 0 0 6pt; }}
.codefile {{ font: bold 9.5pt "Times New Roman", serif; margin: 10pt 0 2pt; border-bottom: 0.6px solid #000;
  break-after: avoid; }}
.codefile span {{ font-weight: normal; color: #555; }}
.code pre {{ margin: 0; font: 6.9pt/1.32 Consolas, "Courier New", monospace; white-space: pre-wrap;
  word-break: break-all; }}
.linenos {{ color: #aaa; }}
</style></head><body>
<h1>Source Code: A Reproduction Study of ToFe</h1>
<p>Hasan Berk Karabay (N25220154), CMP684 Neural Networks, midterm reading project. Reproduction of H. Zhang et al.,
“ToFe: Lagged Token Freezing and Reusing for Efficient Vision Transformer Inference,” IEEE TNNLS, 2026. All files
were written for this project and are listed in order of importance; <code>tofe.py</code> contains the method.</p>
{code}
</body></html>"""


def breakable(url):
    """Link that may wrap only after a "/", never at the hyphens of the URL itself."""
    parts = url.split('/')
    text = '/<wbr>'.join(f'<span style="white-space:nowrap">{p}</span>' for p in parts)
    return f'<a href="{url}" style="color:inherit;text-decoration:none">{text}</a>'


def render(html, name, out, title):
    src = os.path.join(IEEE, f'_{name}.html')
    tmp = os.path.join(IEEE, f'_{name}.pdf')
    open(src, 'w', encoding='utf-8').write(html)
    subprocess.run([EDGE, '--headless=new', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={tmp}',
                    pathlib.Path(src).as_uri()], check=True, timeout=300)
    writer = pypdf.PdfWriter(clone_from=tmp)
    writer.add_metadata({'/Title': title, '/Author': 'Hasan Berk Karabay'})
    with open(out, 'wb') as f:
        writer.write(f)
    print(f'{out}: {len(writer.pages)} pages')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code-url', default=PROJECT_URL, help='project repository, printed under the author block '
                    'and cited as reference [9]')
    args = ap.parse_args()
    html = paper_tables.fill(open(os.path.join(IEEE, 'paper.html'), encoding='utf-8').read())
    link = f'<a href="{args.code_url}" style="color:inherit;text-decoration:none">{args.code_url}</a>'
    html = html.replace('<!--CODE_LINK-->', f'<div class="mail">Code and models: {link}</div>')
    html = html.replace('{{PROJECT_URL}}', breakable(args.code_url))
    html = re.sub(r'\{\{URL:(.+?)\}\}', lambda m: breakable(m.group(1)), html)
    assert '<!--TABLE:' not in html, 'unfilled table placeholder'
    render(html, 'paper', REPORT_PDF, 'A Reproduction Study of ToFe')
    render(CODE_PAGE.format(code=code_appendix()), 'code', CODE_PDF, 'Source code: A Reproduction Study of ToFe')


if __name__ == '__main__':
    main()
