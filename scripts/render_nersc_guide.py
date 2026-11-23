"""Render this simple OOXML runbook using bundled ReportLab and canonical Poppler QA.

The Windows Codex runtime does not ship LibreOffice. The canonical renderer was
attempted and failed for missing soffice.exe. This fallback supports the runbook's
plain paragraphs, monospace code and explicit page breaks, using its actual DOCX.
It is a layout preview, not an assertion of Word/LibreOffice pagination identity.
"""
from pathlib import Path
import importlib.util
import json
import os
from xml.sax.saxutils import escape

from docx import Document
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Preformatted, PageBreak

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path("C:/Users/weita/.cache/codex-runtimes/codex-primary-runtime")
CANONICAL = Path("C:/Users/weita/.codex/plugins/cache/openai-primary-runtime/documents/26.1119.11507/skills/documents/render_docx.py")
OUT = ROOT / "results/nersc_doc_qa/preview"
OUT.mkdir(parents=True, exist_ok=True)


def convert(doc_path, user_profile, convert_tmp_dir, stem, verbose):
    doc = Document(doc_path)
    fonts = Path("C:/Windows/Fonts")
    for name, filename in [("Arial", "arial.ttf"), ("ArialBold", "arialbd.ttf"), ("CourierNew", "cour.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    styles = {
        "Title": ParagraphStyle("title", fontName="Arial", fontSize=21, leading=25.2, spaceAfter=6),
        "Heading 1": ParagraphStyle("h1", fontName="ArialBold", fontSize=16, leading=19.2, spaceBefore=10, spaceAfter=6),
        "Heading 2": ParagraphStyle("h2", fontName="ArialBold", fontSize=11, leading=13.2, spaceBefore=8, spaceAfter=6),
        "Normal": ParagraphStyle("body", fontName="Arial", fontSize=10, leading=11.6, spaceAfter=6),
        "Shell Code": ParagraphStyle("code", fontName="CourierNew", fontSize=8.2, leading=9.6, leftIndent=5.76, spaceBefore=2, spaceAfter=7),
    }
    story = []
    for p in doc.paragraphs:
        if p._p.xpath('.//w:br[@w:type="page"]'):
            story.append(PageBreak());continue
        if not p.text: continue
        style = styles.get(p.style.name, styles["Normal"])
        if p.style.name == "Shell Code":
            story.append(Preformatted(p.text, style, maxLineLength=100, splitChars=" ", newLineChars=""))
        else:
            story.append(Paragraph(escape(p.text).replace("\n", "<br/>"), style))
    pdf = Path(convert_tmp_dir) / (stem + ".pdf")
    def footer(canvas, document):
        canvas.setFont("Arial", 8)
        canvas.drawRightString(565.2, 24, f"NERSC A100 workflow  |  {document.page}")
    SimpleDocTemplate(str(pdf), pagesize=(612, 792), leftMargin=46.8, rightMargin=46.8,
                      topMargin=44.64, bottomMargin=44.64).build(story, onFirstPage=footer, onLaterPages=footer)
    return str(pdf), "Bundled ReportLab OOXML preview; explicit page breaks and literal code preserved."


def main():
    spec = importlib.util.spec_from_file_location("canonical_render", CANONICAL)
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    renderer.convert_to_pdf = convert
    os.environ["PATH"] = str(RUNTIME / "dependencies/native/poppler/Library/bin") + os.pathsep + os.environ.get("PATH", "")
    source = ROOT / "deliverables/NERSC_A100_Training_and_Inference.docx"
    renderer.rasterize(str(source), str(OUT), 120, verbose=True, emit_pdf=True)
    pages = sorted(OUT.glob("page-*.png"))
    report = {"source": str(source), "pages": len(pages), "renderer": "ReportLab OOXML preview plus canonical render_docx.py Poppler rasterization",
              "limitation": "Windows runtime has no bundled LibreOffice; native Word pagination not independently verified"}
    (OUT.parent / "render_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__": main()
