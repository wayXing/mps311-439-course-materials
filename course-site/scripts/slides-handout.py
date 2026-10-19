"""Lay a one-slide-per-page deck out as an A4 handout: three slides per page, ruled
lines on the right for handwritten notes. Vector throughout (slides are not rasterised).

    python slides-handout.py deck.pdf handout.pdf "Header text"
"""
import sys

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

A4_W, A4_H = 595.28, 841.89
MARGIN = 40
PER_PAGE = 3
SLIDE_W = 330                 # points; a 16:9 slide about 11.6 cm wide
INSET = 5                     # white gap between slide edge and its frame
LINE_GAP = 18                 # ruled-line spacing for handwriting
TOP, BOTTOM = A4_H - 44, 40   # band between header and footer
SLOT_H = (TOP - BOTTOM) / PER_PAGE


def esc(text):
    return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')


def slide_y(i, h):
    """Bottom edge of slide i on the page, centred in its slot."""
    slot_top = TOP - i * SLOT_H
    return slot_top - (SLOT_H - h) / 2 - h


def furniture(header, page_no, total, n_slides, h):
    """Header, footer, slide frames and ruled lines, as PDF drawing operators."""
    ops = ['q 0.6 0.6 0.6 RG 0.5 w']
    x0 = MARGIN + SLIDE_W + INSET + 14
    for i in range(n_slides):
        slot_top = TOP - i * SLOT_H
        y = slot_top - 12
        while y > slot_top - SLOT_H + 12:
            ops.append(f'{x0:.2f} {y:.2f} m {A4_W - MARGIN:.2f} {y:.2f} l S')
            y -= LINE_GAP
        ops.append(f'{MARGIN - INSET} {slide_y(i, h) - INSET:.2f} {SLIDE_W + 2 * INSET} {h + 2 * INSET:.2f} re S')
    ops.append('Q')
    ops.append('BT /F1 7.5 Tf 0.45 0.45 0.45 rg')
    ops.append(f'1 0 0 1 {MARGIN} {A4_H - 26} Tm ({esc(header)}) Tj')
    ops.append(f'1 0 0 1 {A4_W - MARGIN - 40} 22 Tm (p. {page_no} / {total}) Tj')
    ops.append('ET')
    return '\n'.join(ops).encode('latin-1', 'replace')


def main(src, out, header):
    deck = PdfReader(src).pages
    sw, sh = float(deck[0].mediabox.width), float(deck[0].mediabox.height)
    scale = SLIDE_W / sw
    h = sh * scale
    total = (len(deck) + PER_PAGE - 1) // PER_PAGE
    font = DictionaryObject({
        NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'),
        NameObject('/BaseFont'): NameObject('/Helvetica'),
    })
    writer = PdfWriter()
    for p in range(total):
        group = deck[p * PER_PAGE:(p + 1) * PER_PAGE]
        page = writer.add_blank_page(A4_W, A4_H)
        stream = DecodedStreamObject()
        stream.set_data(furniture(header, p + 1, total, len(group), h))
        page[NameObject('/Contents')] = writer._add_object(stream)
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): font}),
        })
        for i, slide in enumerate(group):
            page.merge_transformed_page(
                slide, Transformation().scale(scale).translate(MARGIN, slide_y(i, h)), over=False)
    writer.compress_identical_objects()
    with open(out, 'wb') as fh:
        writer.write(fh)


if __name__ == '__main__':
    main(*sys.argv[1:4])
