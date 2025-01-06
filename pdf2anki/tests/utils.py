import math
from typing import List, Optional
from pdf2anki.elements import CharInfo, LineInfo, PageInfo, ParagraphInfo
import json

from pdf2anki.extraction import extract_line_info, extract_page_info, extract_paragraph_info

with open('/home/rookslog/pdf2anki/pdf2anki/tests/test_files/font_dict.json', 'r') as f:
    FONT_DICT = json.load(f)

PAGE_WIDTH = 400.0
PAGE_HEIGHT = 650.0
FONT_WIDTH_SCALING_FACTOR = 800

def compare_float_sets(set1: set, set2: set, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
    if len(set1) != len(set2):
        return False

    list1 = sorted(set1)
    list2 = sorted(set2)

    for a, b in zip(list1, list2):
        if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
            return False

    return True

def insert_headers(text_blocks: list[str], headers: list[str]) -> list[str]:
    if headers is None:
        return text_blocks
    else:
        assert len(headers) == len(text_blocks), "The number of headers must match the number of text blocks."
    text_blocks_with_headers = []
    for header, text_block in zip(headers, text_blocks):
        text_blocks_with_headers.append((header, True))
        text_blocks_with_headers.append((text_block, False))
    return text_blocks_with_headers

def insert_headers_at_index(text_blocks: list[str], headers: list[str]) -> list[str]:
    # text_blocks is given in format [page1text, page2text, ...] with each page text in format "line1\nline2\nline3\n\nline1\nline2\nline3\n\n"
    # headers is a list of tuples (header, index) where index is the index of the paragraph the header belongs to

    text_blocks_with_headers = []
    for i, text_block in enumerate(text_blocks):
        paragraphs = [(paragraph, False) for paragraph in text_block.split('\n\n')]
        if headers is not None:
            header = headers[i]
            paragraphs.insert(header[1], (header[0], True))
        text_blocks_with_headers.append(paragraphs)
    return text_blocks_with_headers # return in format [[(paragraph1, is_header), (paragraph2, is_header), ...], [(paragraph1, is_header), ...], ...]
                                    # each element is a page, each page is a list of tuples (paragraph, is_header)

def create_mock_char(text: str, x: float, y: float, font_size: float=12.0, font: str="Arial", color: str="#000000") -> CharInfo:
    char_width = FONT_DICT[font][text][0] * font_size
    char_height = FONT_DICT[font][text][1] * font_size
    return CharInfo(
        text=text,
        bbox=(x, y-char_height, x+char_width, y),
        size=font_size,
        width=char_width,
        font=font,
        color=color
    )

#TODO: Implement this function
def create_test_doc_todo(text_blocks: list[str], headers: Optional[list[str]], lines_per_page: Optional[int]=None, text_size: Optional[float]=12.0, header_size: Optional[float]=16.0, font_name: Optional[str]="Arial") -> List[PageInfo]:
    if lines_per_page is None: # all in one page
        lines_per_page = len(text_blocks)
    
    text_blocks_with_headers = insert_headers(text_blocks, headers)
    all_lines = []
    for text_block, is_header in text_blocks_with_headers:
        paragraphs = text_block.split('\n\n')
        lines = [(line, i, is_header) for i, line in enumerate(paragraphs)]
        all_lines.extend(lines)
    
    doc = []
    
    page_num = 1
    current_paragraph_num = 0
    last_paragraph_num = None

    while len(all_lines) > 0: # WHILE THERE ARE LINES LEFT
        char_y_pos = PAGE_HEIGHT
        current_page_paragraphs = []
        current_lines = []
        END_OF_PAGE = False
        while not END_OF_PAGE and len(all_lines) > 0: # WHILE WE ARE ON A PAGE AND THERE IS SPACE LEFT
            line, paragraph_num, is_header = all_lines.pop(0)
            if paragraph_num != current_paragraph_num: # NEW PARAGRAPH, SO GENERATE INFO AND START NEW
                paragraph_info = extract_paragraph_info(lines, pagenum=page_num)
                current_page_paragraphs.append(paragraph_info)
                last_paragraph_num = current_paragraph_num
                current_paragraph_num = paragraph_num
                current_lines = []
            char_infos = []
            char_x_pos = 0.0
            for char in line:
                if is_header:
                    char_info = create_mock_char(char, char_x_pos, char_y_pos, header_size, font_name)
                else:
                    char_info = create_mock_char(char, char_x_pos, char_y_pos, text_size, font_name)
                char_infos.append(char_info)
                char_x_pos += char_info.width

            line_info = extract_line_info(char_infos)

            if char_y_pos - line_info.char_height > 0: # IF THERE IS SPACE LEFT ON THE PAGE
                current_lines.append(line_info)
                char_y_pos -= line_info.char_height
                last_paragraph_num = paragraph_num

            else: # IF THERE IS NO SPACE LEFT ON THE PAGE, START NEW PAGE
                paragraph_info = extract_paragraph_info(current_lines, pagenum=page_num)
                paragraph_info.split_end_line = last_paragraph_num == current_paragraph_num
                current_page_paragraphs.append(paragraph_info)
                page_info = extract_page_info(current_page_paragraphs)
                doc.append(page_info)

                page_num += 1
                current_page_paragraphs = []
                current_lines = [line_info]
                END_OF_PAGE = True

        if len(all_lines) == 0:
            pass

    return doc

def create_test_paragraph(text: str, font_size: float=12.0, font_name: str="Arial", start_x: float=0.0, start_y: float=0.0) -> ParagraphInfo:
    lines = text.split('\n')
    char_y_pos = start_y
    lines_info = []
    for line in lines:
        char_x_pos = start_x
        char_infos = []
        for char in line:
            char_info = create_mock_char(char, char_x_pos, char_y_pos, font_size, font_name)
            char_infos.append(char_info)
            char_x_pos += char_info.width
        line_info = extract_line_info(char_infos)
        lines_info.append(line_info)
        char_y_pos -= line_info.char_height
    paragraph_info = extract_paragraph_info(lines_info)
    return paragraph_info


def create_test_doc(text_blocks: list[str], headers: Optional[list[str]], text_size: Optional[float]=12.0, header_size: Optional[float]=16.0, font_name: Optional[str]="Arial") -> List[PageInfo]:
    # text_blocks is given in format [page1text, page2text, ...] with each page text in format "line1\nline2\nline3\n\nline1\nline2\nline3\n\n"
    # headers is given in format [(header1, 1), (header2, 0), ...] with each header element a tuple of the header text + which paragraph it belongs to
    text_blocks_with_headers = insert_headers_at_index(text_blocks, headers)
    doc = []
    for page in text_blocks_with_headers:
        paragraph_info_list = []
        for paragraph, is_header in page:
            if is_header:
                paragraph_info = create_test_paragraph(paragraph, font_size=header_size, font_name=font_name)
            else:
                paragraph_info = create_test_paragraph(paragraph, font_size=text_size, font_name=font_name)
            paragraph_info_list.append(paragraph_info)
        page_info = extract_page_info(paragraph_info_list)
        doc.append(page_info)
    return doc


from pdfminer.layout import LTChar, LTTextLineHorizontal, LTPage, LTFigure, LTContainer
from pdfminer.pdffont import PDFFont
from pdfminer.pdfcolor import PDFColorSpace

class MockPDFGraphicState:
        def __init__(self):
            pass  # Add attributes if necessary


def update_bbox(lt_text_line: LTTextLineHorizontal, new_bbox: tuple) -> None:
    old_bbox = lt_text_line.bbox
    dx = new_bbox[0] - old_bbox[0]
    dy = new_bbox[1] - old_bbox[1]

    # Update the bbox of LTTextLineHorizontal
    lt_text_line.set_bbox(new_bbox)

    # Update the bbox of each contained LTChar
    for char in lt_text_line:
        if isinstance(char, LTChar):
            char_bbox = char.bbox
            new_char_bbox = (
                char_bbox[0] + dx,
                char_bbox[1] + dy,
                char_bbox[2] + dx,
                char_bbox[3] + dy,
            )
            char.set_bbox(new_char_bbox)

def create_mock_ltchar(text: str, x: float, y: float, font_size: float=12.0, font: str="Arial", color: str="DeviceGray") -> LTChar:
    font_descriptor = {"FontName": font}
    
    widths = {char: dimensions[0] * FONT_WIDTH_SCALING_FACTOR for char, dimensions in FONT_DICT[font].items()}
    pdffont = PDFFont(descriptor=font_descriptor, widths=widths)
    pdffont.fontname = font
    ncs = PDFColorSpace(name=color, ncomponents=1)
    char = LTChar(
        matrix=(1, 0, 0, 1, 0, 0),
        font=pdffont,
        fontsize=font_size,
        scaling=1.0,
        rise=0.0,
        text=text,
        textwidth=pdffont.char_width(text),
        textdisp=0.0,
        ncs=ncs,
        graphicstate=MockPDFGraphicState()
    )
    char.set_bbox((x, y - char.height, x + char.width, y))
    return char

def create_mock_lttextline(text: str, x: float, y: float, font_size: float=12.0, font: str="Arial", color: str="DeviceGray") -> LTTextLineHorizontal:
    lttextline = LTTextLineHorizontal(word_margin=0.5)
    for char in text:
        ltchar = create_mock_ltchar(char, x, y, font_size, font, color)
        lttextline.add(ltchar)
        x += ltchar.width
    return lttextline

def create_mock_ltpage(pageid: int, text_block_with_header: list[str], text_size: float=12.0, header_size: float=16.0, font_name: str="Arial") -> LTPage:
    ltpage = LTPage(pageid=pageid, bbox=(0, 0, PAGE_WIDTH, PAGE_HEIGHT))
    y = PAGE_HEIGHT
    space_width = FONT_DICT[font_name][' '][0] * text_size
    for paragraph, is_header in text_block_with_header:
        for i, line in enumerate(paragraph.split('\n')):
            font_size = header_size if is_header else text_size
            lttextline = create_mock_lttextline(line, 0, y, font_size, font_name)
            if is_header:
                x_coord = PAGE_WIDTH / 2 - lttextline.width / 2
                update_bbox(lttextline, (x_coord, y - header_size, PAGE_WIDTH-x_coord, y))
            elif i == 0: # for paragraph indent
                x_coord = 5 * space_width
                update_bbox(lttextline, (x_coord, y - text_size, x_coord + lttextline.width, y))
            ltpage.add(lttextline)
            y -= font_size * 2 if is_header else font_size * 1.2
    return ltpage

def create_test_lt_doc(text_blocks: list[str], headers: Optional[list[str]], text_size: Optional[float]=12.0, header_size: Optional[float]=16.0, font_name: Optional[str]="Arial") -> List[LTPage]:
    """
    Create a list of mock LTPage elements based on the given text blocks and headers.

    Args:
        text_blocks (list[str]): List of text blocks for each page.
        headers (list[str]): List of headers with their positions.
        text_size (float): Font size for the text.
        header_size (float): Font size for the headers.
        font_name (str): Font name.

    Returns:
        List[LTPage]: List of mock LTPage elements.
    """
    ltpages = []
    text_blocks_with_headers = insert_headers_at_index(text_blocks, headers)
    for pageid, text_block_with_header in enumerate(text_blocks_with_headers, start=1):
        ltpage = create_mock_ltpage(pageid, text_block_with_header, text_size, header_size, font_name)
        ltpages.append(ltpage)
    return ltpages