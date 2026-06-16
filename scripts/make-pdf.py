#!/usr/bin/env python3
"""
make-pdf.py  —  Converte screenshots PNG de slides em um único PDF.

Uso: python3 scripts/make-pdf.py <slides_dir> <output.pdf>
Ex:  python3 scripts/make-pdf.py /tmp/slides-lec01 dist/lec01/lec01.pdf
"""

import sys
import glob
import img2pdf

slides_dir = sys.argv[1]
out_pdf    = sys.argv[2]

slides = sorted(glob.glob(f'{slides_dir}/slide-*.png'))

if not slides:
    print(f'ERRO: Nenhum screenshot encontrado em {slides_dir}!')
    sys.exit(1)

with open(out_pdf, 'wb') as f:
    f.write(img2pdf.convert(slides))

print(f'PDF criado: {len(slides)} páginas -> {out_pdf}')
