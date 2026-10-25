"""PaddleOCR PP-StructureV3, table recognition on. Run with the venv that has paddleocr:

    /home/ubuntu/tmp/paddle-venv/bin/python runners/paddleocr.py
"""
from paddleocr import PPStructureV3

from _common import page_paths, write


def main() -> None:
    pipeline = PPStructureV3(
        layout_detection_model_name="PP-DocLayout-S",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        use_table_recognition=True,
        use_formula_recognition=False,
        use_chart_recognition=False,
        enable_mkldnn=False,
    )
    pages = {}
    for page in page_paths():
        result = pipeline.predict(str(page))[0]
        markdown = result.markdown
        pages[page.stem] = markdown.get("markdown_texts", "") if isinstance(markdown, dict) else str(markdown)
        print(f"{page.stem}: {len(pages[page.stem])} chars", flush=True)
    write("paddleocr-structure", pages)


if __name__ == "__main__":
    main()
