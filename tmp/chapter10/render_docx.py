"""Run the document skill's canonical PNG renderer using native Word's PDF export.

Native Word is available on this host; LibreOffice is not. The PDF is exported
from the saved final DOCX immediately before this call. The adapter changes only
the PDF conversion backend and retains the packaged rasterize/CLI implementation.
"""
from pathlib import Path
import importlib.util, shutil, sys

skill=Path(r'C:\Users\sinam\.codex\plugins\cache\openai-primary-runtime\documents\26.903.11726\skills\documents\render_docx.py')
spec=importlib.util.spec_from_file_location('canonical_render_docx',skill)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
pdf=Path(__file__).with_name('final_word.pdf')

def word_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose):
    assert Path(doc_path).stat().st_mtime <= pdf.stat().st_mtime, 'PDF is older than DOCX'
    out=Path(convert_tmp_dir)/(stem+'.pdf');shutil.copy2(pdf,out)
    return str(out),'Native Microsoft Word ExportAsFixedFormat'

mod.convert_to_pdf=word_pdf
mod.main()
