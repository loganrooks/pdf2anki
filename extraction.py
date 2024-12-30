from dataclasses import dataclass
import logging
import os
import time
from typing import Dict, List, Optional, Set, Tuple
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTPage, LAParams, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal, LTFigure
from utils import log_time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



@dataclass
class LineInfo:
    text: str
    chars: List[dict]
    bbox: Tuple[float]
    size: float
    font: str
    color: str
    char_width: float
    char_height: float
    split_end_word: bool

@dataclass
class CharInfo:
    text: str
    bbox: Tuple[float]
    size: float
    font: str
    color: str
    height: float
    width: float

@dataclass
class ParagraphInfo:
    text: str
    lines: List[LineInfo]
    bbox: Tuple[float]
    fonts: Set[str]
    colors: Set[str]
    char_width: float
    size: float
    split_end_line: bool
    is_indented: bool

@dataclass
class PageInfo:
    text: str
    bbox: Tuple[float]
    fonts: Set[str]
    sizes: Set[float]
    colors: Set[str]
    paragraphs: List[ParagraphInfo]
    split_end_paragraph: bool
    pagenum: int

def get_average(numbers):
    """
    Calculate the average of a list of numbers.

    :param numbers: List of numbers
    :return: Average of the numbers
    """
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def concat_bboxes(bboxes: List[Tuple[float]]) -> Tuple[float]:
    """
    Concatenate a list of bounding boxes into a single bounding box.

    Args:
        bboxes (list): A list of bounding boxes.

    Returns:
        tuple: A single bounding box that encompasses all the input bounding boxes.
    """
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[2] for bbox in bboxes)
    y1 = max(bbox[3] for bbox in bboxes)
    return (x0, y0, x1, y1)

def contained_in_bbox(bbox1: Tuple[float], bbox2: Tuple[float], bbox_overlap: float = 1.0) -> bool:
    """
    Check if bbox1 is contained in bbox2 based on the overlap percentage.

    Args:
        bbox1 (tuple): Bounding box 1.
        bbox2 (tuple): Bounding box 2.
        bbox_overlap (float): Overlap percentage of bbox1's area that must be in bbox2.

    Returns:
        bool: True if bbox1 is contained in bbox2, False otherwise.
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Calculate the area of bbox1
    area1 = (x2 - x1) * (y2 - y1)

    # Calculate the intersection area
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        intersection_area = 0

    return intersection_area >= bbox_overlap * area1
    
    

def extract_char_info(ltchar: LTChar) -> CharInfo:
    """
    Convert an LTChar object to a CharInfo dataclass.

    Args:
        ltchar (LTChar): The LTChar object to convert.

    Returns:
        CharInfo: A CharInfo dataclass with equivalent information.
    """
    return CharInfo(
        text=ltchar.get_text(),
        bbox=ltchar.bbox,
        size=ltchar.size,
        font=ltchar.fontname,
        color=ltchar.ncs.name,
        height=ltchar.height,
        width=ltchar.width
    )

@log_time
def extract_text_from_figure(figure, clip: Optional[Tuple[float]] = None) -> List[CharInfo]:
    text_elements = []
    for element in figure:
        if isinstance(element, LTChar):
            if clip is None or contained_in_bbox(element.bbox, clip):
                text_elements.append(extract_char_info(element))
        elif isinstance(element, (LTTextBoxHorizontal, LTTextLineHorizontal, LTFigure)):
            text_elements.extend(extract_text_from_figure(element, clip))
    return text_elements

def extract_line_info(line: List[CharInfo], interrupt_chars: str = "-") -> LineInfo:
    """
    Extract information from a list of CharInfo elements that form a line.

    Args:
        line (list): A list of CharInfo elements that form a line.

    Returns:
        LineInfo: A LineInfo dataclass with information about the line.
    """
    line_text = "".join(char.text for char in line)

    return LineInfo(
        text=line_text,
        chars=line,
        bbox=concat_bboxes([char.bbox for char in line]),
        size=get_average([char.size for char in line]),
        font=line[0].font,
        color=line[0].color,
        char_width=get_average([char.width for char in line]),
        char_height=get_average([char.height for char in line]),
        split_end_word=line_text.rstrip().endswith(interrupt_chars) and not line_text.rstrip().removesuffix(interrupt_chars)[-1].isspace(),
    )

def extract_word_info(word: List[CharInfo], separator: Optional[str] = " ") -> dict:
    """
    Extract information from a list of CharInfo elements that form a word.

    Args:
        word (list): A list of CharInfo elements that form a word.

    Returns:
        dict: A dictionary with information about the word.
    """
    word_text = "".join(char.text for char in word)
    return {
        "text": word_text,
        "bbox": concat_bboxes([char.bbox for char in word]),
        "size": get_average([char.size for char in word]),
        "font": word[0].font,
        "color": word[0].color,
        "char_width": get_average([char.width for char in word]),
        "is_upper": word_text.strip().isupper()
    }

def extract_words_from_text(text_elements: List[CharInfo], separator: Optional[str] = " ") -> List[dict]:
    words = []
    current_word = []
    last_char_separator = False

    for char in text_elements:
        if char.text == separator:
            last_char_separator = True
            current_word.append(char)
        elif last_char_separator:
            word_info = extract_word_info(current_word)
            words.append(word_info)
            current_word = [char]
        else:
            current_word.append(char)

    if current_word:
        word_info = extract_word_info(current_word)
        words.append(word_info)

    return words

def get_y_overlap(bbox1: Tuple[float], bbox2: Tuple[float]) -> float:
    """
    Calculate the vertical overlap between two bounding boxes.

    Args:
        bbox1 (tuple): Bounding box 1.
        bbox2 (tuple): Bounding box 2.

    Returns:
        float: The vertical overlap between the two bounding boxes.
    """
    y1, y2 = bbox1[1], bbox1[3]
    y3, y4 = bbox2[1], bbox2[3]
    return min(y2, y4) - max(y1, y3)

@log_time
def extract_lines_from_text_elements(text_elements: List[CharInfo], char_margin_factor: float = 0.5, line_overlap_factor: float = 0.5, interrupt_chars: str = "-") -> List[LineInfo]:
    lines = []
    current_line = []
    last_element = None

    for element in text_elements:
        char_margin = element.width * char_margin_factor
        min_overlap = element.height * line_overlap_factor

        if last_element is None:
            current_line.append(element)
            last_element = element
        else:
            y_overlap = abs(get_y_overlap(element.bbox, last_element.bbox))
            x_gap = abs(last_element.bbox[2] - element.bbox[0])

            if y_overlap >= min_overlap and x_gap <= char_margin:
                current_line.append(element)
                last_element = element
            else:
                current_line_info = extract_line_info(current_line, interrupt_chars=interrupt_chars)
                lines.append(current_line_info)
                current_line = [element]
                last_element = element

    if current_line:
        current_line_info = extract_line_info(current_line)
        lines.append(current_line_info)

    return lines

def extract_lines_from_figure(figure, char_margin_factor: float = 0.5, clip: Optional[Tuple[float]] = None) -> List[LineInfo]:
    text_elements = extract_text_from_figure(figure, clip=clip)
    return extract_lines_from_text_elements(text_elements, char_margin_factor=char_margin_factor)

def is_indented(paragraph_info: ParagraphInfo, indent_factor: float = 3.0) -> bool:
    """
    Check if a paragraph is indented.

    Args:
        paragraph_info (ParagraphInfo): Information about the paragraph.

    Returns:
        bool: True if the paragraph is indented, False otherwise.
    """
    indent = paragraph_info.lines[0].bbox[0] - paragraph_info.bbox[0]
    return indent >= indent_factor * paragraph_info.char_width

def is_indented(line_a: LineInfo, line_b: LineInfo, indent_factor: float = 3.0) -> bool:
    """
    Check if one line is indented with respect to another.

    Args:
        line_a (LineInfo): Information about the first line.
        line_b (LineInfo): Information about the second line.

    Returns:
        bool: True if line_b is indented with respect to line_a, False otherwise.
    """
    indent = line_b.bbox[0] - line_a.bbox[0]
    return indent >= indent_factor * line_a.char_width

def extract_paragraph_info(paragraph: List[LineInfo], indent_factor: float = 3.0) -> ParagraphInfo:
    """
    Extract information from a list of LineInfo elements that form a paragraph.

    Args:
        paragraph (list): A list of LineInfo elements that form a paragraph.

    Returns:
        ParagraphInfo: A ParagraphInfo dataclass with information about the paragraph.
    """
    paragraph_text = "".join(line.text for line in paragraph)
    return ParagraphInfo(
        text=paragraph_text,
        lines=paragraph,
        bbox=concat_bboxes([line.bbox for line in paragraph]),
        fonts=set([line.font for line in paragraph]),
        colors = set([line.color for line in paragraph]),
        char_width=get_average([line.char_width for line in paragraph]),
        size=get_average([line.size for line in paragraph]),
        split_end_line=paragraph[-1].split_end_word,
        is_indented=is_indented(paragraph[1], paragraph[0], indent_factor=indent_factor) if len(paragraph) > 1 else False
    )

def extract_page_info(page: List[ParagraphInfo], pagenum: int) -> PageInfo:
    """
    Extract information from a list of ParagraphInfo elements that form a page.

    Args:
        page (list): A list of ParagraphInfo elements that form a page.

    Returns:
        PageInfo: A PageInfo dataclass with information about the page.
    """
    page_text = "\n\n".join(paragraph.text for paragraph in page)
    return PageInfo(
        text=page_text,
        bbox=concat_bboxes([paragraph.bbox for paragraph in page]),
        fonts=set([font for paragraph in page for font in paragraph.fonts]),
        sizes=set([size for paragraph in page for size in paragraph.size]),
        colors=set([color for paragraph in page for color in paragraph.colors]),
        paragraphs=page,
        split_end_paragraph=page[-1].split_end_line,
        pagenum=pagenum
    )
        


@log_time
def extract_paragraphs_from_page(page: LTPage, 
                                 char_margin_factor: float = 0.5, 
                                 line_margin_factor: float = 0.5, 
                                 clip: Optional[Tuple[float]] = None, 
                                 bbox_overlap: float = 1.0,
                                 indent_factor: float = 3.0) -> List[PageInfo]:
    paragraphs = []
    lines = extract_lines_from_figure(page, char_margin_factor=char_margin_factor)
    current_paragraph = []
    last_line = None

    for line in lines:
        if clip is None or (clip is not None and contained_in_bbox(line.bbox, clip, bbox_overlap=bbox_overlap)):
            if last_line is None:
                current_paragraph.append(line)
                last_line = line
            else:
                line_gap = last_line.bbox[1] - line.bbox[3]
                max_line_gap = line.char_height * line_margin_factor
                
                if line_gap <= max_line_gap and not is_indented(last_line, line, indent_factor=indent_factor):
                    current_paragraph.append(line)
                    last_line = line
                else:
                    current_paragraph_info = extract_paragraph_info(current_paragraph, indent_factor=indent_factor)
                    paragraphs.append(current_paragraph_info)
                    current_paragraph = [line]
                    last_line = line

    if current_paragraph:
        paragraph_info = extract_paragraph_info(current_paragraph, indent_factor=indent_factor)
        paragraphs.append(paragraph_info)

    return paragraphs

@log_time
def process_ltpages(doc: List[LTPage], char_margin_factor: float = 0.5, line_margin_factor: float = 0.5, margins: Optional[Tuple[float]] = None, bbox_overlap: float = 1.0) -> List[PageInfo]:
    pages = []
    for page_num, page in enumerate(doc, start=1):
        clip = (margins[0], margins[1], page.width - margins[2], page.height - margins[3]) if margins else None
        paragraphs = extract_paragraphs_from_page(page, char_margin_factor=char_margin_factor, line_margin_factor=line_margin_factor, clip=clip, bbox_overlap=bbox_overlap)
        page_info = extract_page_info(paragraphs, pagenum=page_num)
        pages.append(page_info)
    return pages

def main():
    pdf_path = os.getcwd() + "/examples/pathmarks_ocr.pdf"
    params = LAParams(line_overlap=0.7, char_margin=4, line_margin=0.6)
    doc = list(extract_pages(pdf_path, laparams=params, maxpages=80))
    margins = (0, 20, 0, 15)
    bbox_overlap = 0.8
    processed_pages = process_ltpages(doc, char_margin_factor=params.char_margin, line_margin_factor=params.line_margin, margins=margins, bbox_overlap=bbox_overlap)
    for page in processed_pages:
        print(f"Page {page['pagenum']}")
        for paragraph in page["paragraphs"]:
            print(paragraph.text)
        print("\n")

if __name__ == "__main__":
    main()