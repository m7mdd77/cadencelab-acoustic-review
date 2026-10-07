"""Build the reviewed technical markdown as a bounded competition PDF."""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from pypdf import PdfReader


def build():
    root = Path(__file__).resolve().parent
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9.5
    styles['BodyText'].leading = 13
    styles['BodyText'].spaceAfter = 6
    styles['Heading2'].fontSize = 12
    styles['Heading2'].leading = 15
    story = []
    for block in (root / 'TECHNICAL_REPORT.md').read_text(encoding='utf-8').split('\n\n'):
        block = block.strip()
        if not block:
            continue
        style = 'Title' if block.startswith('# ') else 'Heading2' if block.startswith('## ') else 'BodyText'
        text = block.removeprefix('## ').removeprefix('# ')
        story.append(Paragraph(escape(text).replace('\n', '<br/>'), styles[style]))
        if style == 'Title':
            story.append(Spacer(1, 3 * mm))
    output = root / 'CadenceLab-technical-report.pdf'

    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(18 * mm, 12 * mm, 'CadenceLab | AI-assisted prototype | October 7, 2026')
        canvas.drawRightString(192 * mm, 12 * mm, str(doc.page))

    SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm,
                      rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=20 * mm,
                      title='CadenceLab Technical Report').build(story, onFirstPage=footer, onLaterPages=footer)
    pages = len(PdfReader(output).pages)
    if pages > 6:
        raise RuntimeError(f'Report has {pages} pages; competition maximum is six')
    print(f'{output.name}: {pages} pages')


if __name__ == '__main__':
    build()
