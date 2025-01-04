import pdb
from PIL import Image
from typing import Any, Callable, List
import pytest
from pdf2anki.extraction import (
    extract_char_info,
    extract_line_info,
    extract_paragraph_info,
    extract_page_info,
    process_ltpages,
    remove_file_objects,
    restore_file_objects,
    get_values_from_ltpage
)

from pdf2anki.utils import concat_bboxes
from pdf2anki.elements import CharInfo, LineInfo, ParagraphInfo, PageInfo, FileObject, FileType
from pdfminer.layout import LTChar, LTPage, LTTextLineHorizontal, LTFigure, LTImage, LTComponent, LTContainer, LTTextContainer, LTLayoutContainer, LTItemT
from pdfminer.pdffont import PDFFont
from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdfparser import PDFStream, PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import PDFObjRef, DecipherCallable, stream_value
from pdfminer.psparser import literal_name, LIT
import io
import os
from enum import Enum
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# Add mock PDFGraphicState class for testing
class MockPDFGraphicState:
    def __init__(self):
        pass  # Add attributes if necessary

def test_extract_char_info():
    # Initialize required arguments for LTChar
    matrix = (1, 0, 0, 1, 0, 0)  # Identity matrix
    font_descriptor = {"FontName": "TestFont"}
    widths = {ord('A'): 10.0}  # Minimal widths dictionary
    font = PDFFont(descriptor=font_descriptor, widths=widths)  # Properly initialized PDFFont
    font.fontname = "TestFont"  # Manually add the 'fontname' attribute

    fontsize = 10.0
    scaling = 1.0  # Default scaling
    rise = 0.0  # Default rise
    text = 'A'
    textwidth = 1.0
    textdisp = 0.0  # Default text displacement
    ncs = PDFColorSpace(name='DeviceGray', ncomponents=1)  # Properly initialized PDFColorSpace
    graphicstate = MockPDFGraphicState()  # Mock graphic state

    # Create LTChar instance with all required arguments using keyword arguments
    ltchar = LTChar(
        matrix=matrix,
        font=font,
        fontsize=fontsize,
        scaling=scaling,
        rise=rise,
        text=text,
        textwidth=textwidth,
        textdisp=textdisp,
        ncs=ncs,
        graphicstate=graphicstate
    )

    # Extract CharInfo
    char_info = extract_char_info(ltchar)
    
    # Assertions
    assert char_info.text == 'A', "Char text does not match"
    assert char_info.bbox == (0.0, 0.0, 10.0, 10.0), "Char bbox does not match"
    assert char_info.size == 10.0, "Char size does not match"
    assert char_info.font == 'TestFont', "Char font does not match"
    assert char_info.color == 'DeviceGray', "Char color does not match"
    assert char_info.height == 10.0, "Char height does not match"
    assert char_info.width == 10.0, "Char width does not match"

def test_extract_line_info():
    # Mock CharInfo objects
    char1 = CharInfo(text='H', bbox=(0, 0, 10, 10), size=12.0, font='FontA', color='black', height=10.0, width=10.0)
    char2 = CharInfo(text='i', bbox=(10, 0, 18, 10), size=12.0, font='FontA', color='black', height=10.0, width=8.0)
    line = [char1, char2]

    # Extract LineInfo
    line_info = extract_line_info(line)
    
    # Assertions
    assert line_info.text == 'Hi', "Line text does not match"
    assert line_info.bbox == (0, 0, 18, 10), "Line bbox does not match"
    assert line_info.font_size == 12.0, "Line font size does not match"
    assert line_info.fonts == {'FontA'}, "Line fonts do not match"
    assert line_info.colors == {'black'}, "Line colors do not match"
    assert line_info.char_width == 9.0, "Line char width does not match"
    assert line_info.char_height == 10.0, "Line char height does not match"
    assert not line_info.split_end_word, "Line split_end_word should be False"

def test_concat_bboxes():
    bboxes = [(0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10)]
    concatenated = concat_bboxes(bboxes)
    assert concatenated == (0, 0, 30, 10), "Concatenated bbox does not match"

def test_extract_paragraph_info():
    # Mock LineInfo objects
    line1 = LineInfo(
        text="This is a test. ",
        chars=[],
        bbox=(0, 0, 100, 10),
        font_size=12.0,
        fonts={'FontA'},
        colors={'black'},
        char_width=10.0,
        char_height=10.0,
        split_end_word=False
    )
    line2 = LineInfo(
        text="This is a continuation. ",
        chars=[],
        bbox=(0, 10, 120, 20),
        font_size=12.0,
        fonts={'FontA'},
        colors={'black'},
        char_width=10.0,
        char_height=10.0,
        split_end_word=False
    )
    paragraph = [line1, line2]

    # Extract ParagraphInfo
    paragraph_info = extract_paragraph_info(paragraph, pagenum=1, indent_factor=3.0)
    
    # Assertions
    assert paragraph_info.text == "This is a test. This is a continuation. ", "Paragraph text does not match"
    assert paragraph_info.bbox == (0, 0, 120, 20), "Paragraph bbox does not match"
    assert paragraph_info.fonts == {'FontA'}, "Paragraph fonts do not match"
    assert paragraph_info.colors == {'black'}, "Paragraph colors do not match"
    assert paragraph_info.char_width == 10.0, "Paragraph char width does not match"
    assert paragraph_info.font_size == 12.0, "Paragraph font size does not match"
    assert not paragraph_info.split_end_line, "Paragraph split_end_line should be False"
    assert not paragraph_info.is_indented, "Paragraph is_indented should be False"

def test_extract_page_info():
    # Mock ParagraphInfo objects
    paragraph1 = ParagraphInfo(
        pagenum=1,
        text="First paragraph.",
        lines=[],
        bbox=(0, 100, 100, 150),
        fonts={'FontA'},
        colors={'black'},
        char_width=10.0,
        font_size=12.0,
        split_end_line=False,
        is_indented=False
    )
    paragraph2 = ParagraphInfo(
        pagenum=1,
        text="Second paragraph.",
        lines=[],
        bbox=(0, 50, 100, 100),
        fonts={'FontA'},
        colors={'black'},
        char_width=10.0,
        font_size=12.0,
        split_end_line=False,
        is_indented=False
    )
    page = [paragraph1, paragraph2]

    # Extract PageInfo
    page_info = extract_page_info(page, tolerance=0.1)
    
    # Assertions
    assert page_info.text == "First paragraph.\n\nSecond paragraph.", "Page text does not match"
    assert page_info.bbox == (0, 50, 100, 150), "Page bbox does not match"
    assert page_info.fonts == {'FontA'}, "Page fonts do not match"
    assert page_info.font_sizes == [12.0], "Page font sizes do not match"
    assert page_info.char_widths == [10.0], "Page char widths do not match"
    assert page_info.colors == {'black'}, "Page colors do not match"
    assert page_info.paragraphs == page, "Page paragraphs do not match"
    assert not page_info.split_end_paragraph, "Page split_end_paragraph should be False"
    assert not page_info.starts_with_indent, "Page starts_with_indent should be False"


def test_extract_page_info_two_fonts():
    # Mock ParagraphInfo objects
    paragraph1 = ParagraphInfo(
        pagenum=1,
        text="First paragraph.",
        lines=[],
        bbox=(0, 100, 100, 150),
        fonts={'FontA'},
        colors={'black'},
        char_width=12.0,
        font_size=14.0,
        split_end_line=False,
        is_indented=False
    )
    paragraph2 = ParagraphInfo(
        pagenum=1,
        text="Second paragraph.",
        lines=[],
        bbox=(0, 50, 100, 100),
        fonts={'FontB'},
        colors={'red'},
        char_width=10.0,
        font_size=12.0,
        split_end_line=False,
        is_indented=False
    )
    paragraph3 = ParagraphInfo(
        pagenum=1,
        text="Second paragraph.",
        lines=[],
        bbox=(0, 0, 100, 50),
        fonts={'FontB'},
        colors={'red'},
        char_width=10.0,
        font_size=12.0,
        split_end_line=False,
        is_indented=False
    )
    page = [paragraph1, paragraph2, paragraph3]

    # Extract PageInfo
    page_info = extract_page_info(page, tolerance=0.1)
    assert page_info.fonts == {'FontA', 'FontB'}, "Page fonts do not match"
    assert page_info.font_sizes == [12.0, 14.0], "Page font sizes do not match"
    assert page_info.char_widths == [10.0, 12.0], "Page char widths do not match"
    assert page_info.colors == {'black', 'red'}, "Page colors do not match"
