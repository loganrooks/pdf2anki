import argparse
import inspect
import io
import logging
import os
import pdb
import pickle
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, get_args
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTPage, LAParams, LTChar, LTTextBoxHorizontal, LTTextLineHorizontal, LTFigure, LTComponent, LTImage
from pdfminer.pdftypes import PDFStream
from pdf2anki import config
from pdf2anki.utils import get_averages, get_average, concat_bboxes, contained_in_bbox, get_y_overlap
from pdf2anki.decorators import conditional_decorator, count_calls, log_time, progress_monitor
from pdf2anki.elements import CharInfo, FileObject, LineInfo, ParagraphInfo, PageInfo, Primitive, FileType


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# HANDLING SAVING AND LOADING OF PAGES

def get_values_from_ltpage(lt_page: LTPage, file_objects: List[FileObject]) -> List[Any]:
    """
    Given an LTPage and a list of FileObjects, use the path attribute from each FileObject
    to return the value in LTPage at the end of the path.
    """
    values = []
    for file_obj in file_objects:
        current = lt_page
        for attr in file_obj.path:
            current = getattr(current, attr, None)
            if current is None:
                break
        values.append(current)
    return values

def remove_lt_images(page: LTPage) -> LTPage:
    def remove_images_from_figure(figure: LTFigure) -> LTFigure:
        figure._objs = [obj for obj in figure if not isinstance(obj, LTImage)]
        for obj in figure:
            if isinstance(obj, LTFigure):
                remove_images_from_figure(obj)
        return figure

    page._objs = [obj for obj in page if not isinstance(obj, LTImage)]
    for obj in page:
        if isinstance(obj, LTFigure):
            remove_images_from_figure(obj)
    return page

def save_and_remove_images(pages: List[LTPage], filepath: str) -> None:
    assert all(isinstance(page, LTPage) for page in pages), "All elements in the list must be LTPage objects."
    cleaned_pages = [remove_lt_images(page) for page in pages]

    with open(filepath, "wb") as f:
        pickle.dump(cleaned_pages, f)

@conditional_decorator(count_calls(track_args=["obj", "path"]), config.DEBUG)
def remove_file_objects(
    obj: Union[LTComponent, Primitive, FileType],
    path: str = "",
    visited: Optional[Set[int]] = None,
    start: Optional[str] = None,
) -> List[FileObject]:
    """Doing this not as a method to test whether it speeds up pickling"""

    if start is not None and visited is None:
        # Initialize the path string based on the start attribute
        path = start
        # Access the object at the initialized path
        for part in re.split(r'\.|\[|\]', path):
            if part.isdigit():
                obj = obj[int(part)]
            elif part:
                obj = getattr(obj, part)
        start = None

    if visited is None:
        visited = set()
    if id(obj) in visited:
        return []

    visited.add(id(obj))
    removed: List[FileObject] = []

    # If this object itself is a file object
    if isinstance(obj, get_args(FileType)):
        file_obj = FileObject(path, type(obj), obj.name)
        removed.append(file_obj)
        return removed

    # If it has attributes
    elif hasattr(obj, "__dict__"):
        for attr_name, attr_value in obj.__dict__.items():
            subpath = f"{path}.{attr_name}" if path else attr_name
            result = remove_file_objects(obj=attr_value, visited=visited, path=subpath)
            removed.extend(result)

            if result and subpath == removed[-1].path:
                setattr(obj, attr_name, None)

    # If it is an iterable
    elif isinstance(obj, (list, tuple, set)):
        new_items = []
        for idx, item in enumerate(obj):
            subpath = f"{path}[{idx}]"
            result = remove_file_objects(obj=item, visited=visited, path=subpath)
            removed.extend(result)
            if not result or not subpath == removed[-1].path:
                new_items.append(item)
        if isinstance(obj, list):
            obj[:] = new_items
        elif isinstance(obj, tuple):
            obj = tuple(new_items)
        elif isinstance(obj, set):
            obj.clear()
            obj.update(new_items)
    return removed

def restore_file_objects(
        file_objects: List[FileObject], 
        obj: Any,
        ) -> Any:
    """
    Restore file objects to the obj. A more general version of restore_lt_images.

    Args:
    """
    if file_objects is None:
        return obj

    for file_obj in file_objects:
        parts = [part for part in re.split(r'\.|\[|\]', file_obj.path) if part]
        current_obj = obj
        for part in parts[:-1]:
            if part.isdigit():
                # Then its an index and current_obj must be a list
                # We must be careful that the list is not empty and try to index it

                if int(part) >= len(current_obj):
                    pdb.set_trace()
                current_obj = current_obj[int(part)]
            elif part:
                current_obj = getattr(current_obj, part)
        
        last_part = parts[-1]
        if last_part.isdigit():
            
            current_obj[int(last_part)] = file_obj.type(open(file_obj.name, "rb"))
        elif last_part:
            setattr(current_obj, last_part, file_obj.type(open(file_obj.name, "rb")))
    return obj


def save_and_remove_file_objects(pages: List[LTPage], filepath: str, with_file_objects: bool = False) -> None:
    assert all(isinstance(page, LTPage) for page in pages), "All elements in the list must be LTPage objects."
    serializable_pages = []
    all_file_objs: Dict[int, List[FileObject]] = {}

    if config.DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)

    for page in pages:
        removed = remove_file_objects(page, start="._objs[1]._objs[0]")
        all_file_objs[page.pageid] = removed
        serializable_pages.append(page)
    
    if config.DEBUG:
        logging.getLogger().setLevel(logging.INFO)

    with open(filepath, "wb") as f:
        if with_file_objects:
            pickle.dump((serializable_pages, all_file_objs), f)
        else:
            pickle.dump((serializable_pages, None), f)

def load_pages(filepath: str, with_file_objects: bool = False) -> List[LTPage]:
    with open(filepath, "rb") as f:
        pages, all_file_objs = pickle.load(f)

    assert isinstance(pages, Iterable), "The first element in the tuple must be an iterable."
    
    assert isinstance(all_file_objs, (dict, None)), "The second element in the tuple must be a dictionary or None"

    if config.DEBUG:
        logging.getLogger().setLevel(logging.DEBUG)

    if all_file_objs is not None and with_file_objects:
        for page in pages:
            assert isinstance(page, LTPage), "All elements in the list must be SerializableLTPage objects."
            file_objs = all_file_objs[page.pageid]
            restore_file_objects(file_objs, page)

    if all_file_objs is None and with_file_objects:
        logging.warning("File objects are not present in the file. Returning only the LTPage objects without file objects.")

    return pages

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
        font_size=get_average([char.size for char in line]),
        fonts=set([char.font for char in line]),
        colors=set([char.color for char in line]),
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
        if (char.text == separator):
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

def is_centered(line: LineInfo, bbox: Tuple[float], tolerance_factor: float = 0.001) -> bool:
    """
    Check if a line is centered in the bounding box.

    Args:
        line (LineInfo): Information about the line.
        bbox (Tuple[float]): The bounding box that the line must be centered in.
        tolerance_factor (float): The percentage of the page width that the line must be centered within.

    Returns:
        bool: True if the line is centered, False otherwise.
    """
    line_center = (line.bbox[0] + line.bbox[2]) / 2
    bbox_center = (bbox[0] + bbox[2]) / 2
    bbox_width = bbox[2] - bbox[0]
    return abs(line_center - bbox_center) <= tolerance_factor * bbox_width

def is_header_continuation(line_a: LineInfo, line_b: LineInfo, tolerance_factors: List[float] = [0.001, 0.1]) -> bool:
    """
    Check if line_b is a continuation of a header from line_a.

    Args:
        line_a (LineInfo): Information about the first line.
        line_b (LineInfo): Information about the second line.
        tolerance_factors (List[float, float]): Tolerance factors for the centering and character width.

    Returns:
        bool: True if line_b is a continuation of a header from line_a, False otherwise.
    """
    header_font_size = line_a.font_size
    line_b_centered = is_centered(line_b, line_a.bbox, tolerance_factor=tolerance_factors[0])
    similar_font_size = abs(line_b.font_size - header_font_size) <= header_font_size*tolerance_factors[1]
    return line_b_centered and similar_font_size

def extract_paragraph_info(paragraph: List[LineInfo], pagenum: Optional[int] = None, indent_factor: float = 3.0) -> ParagraphInfo:
    """
    Extract information from a list of LineInfo elements that form a paragraph.

    Args:
        paragraph (list): A list of LineInfo elements that form a paragraph.

    Returns:
        ParagraphInfo: A ParagraphInfo dataclass with information about the paragraph.
    """
    paragraph_text = "".join(line.text for line in paragraph)
    return ParagraphInfo(
        pagenum=pagenum,
        text=paragraph_text,
        lines=paragraph,
        bbox=concat_bboxes([line.bbox for line in paragraph]),
        fonts=set(font for line in paragraph for font in line.fonts),
        colors=set(color for line in paragraph for color in line.colors),
        char_width=get_average([line.char_width for line in paragraph]),
        font_size=get_average([line.font_size for line in paragraph]),
        split_end_line=paragraph[-1].split_end_word,
        is_indented=is_indented(paragraph[1], paragraph[0], indent_factor=indent_factor) if len(paragraph) > 1 else False
    )

def extract_page_info(page: List[ParagraphInfo], tolerance: float = 1e-1) -> PageInfo:
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
        fonts=set(font for paragraph in page for font in paragraph.fonts),
        font_sizes=get_averages([paragraph.font_size for paragraph in page], tolerance=tolerance),
        char_widths=get_averages([paragraph.char_width for paragraph in page], tolerance=tolerance),
        colors=set(color for paragraph in page for color in paragraph.colors),
        paragraphs=page,
        split_end_paragraph=page[-1].split_end_line,
        starts_with_indent=page[0].is_indented
    )



@log_time
def extract_paragraphs_from_page(page: LTPage, 
                                 pagenum: Optional[int] = None,
                                 char_margin_factor: float = 4.0, 
                                 line_margin_factor: float = 0.5, 
                                 clip: Optional[Tuple[float]] = None, 
                                 bbox_overlap: float = 1.0,
                                 indent_factor: float = 3.0) -> List[PageInfo]:
    paragraphs = []
    lines = extract_lines_from_figure(page, char_margin_factor=char_margin_factor)
    if not lines:
        return None
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
                
                if line_gap > max_line_gap or (is_indented(last_line, line, indent_factor=indent_factor) and not is_header_continuation(last_line, line)):
                    current_paragraph_info = extract_paragraph_info(current_paragraph, pagenum=pagenum, indent_factor=indent_factor)
                    paragraphs.append(current_paragraph_info)
                    current_paragraph = [line]
                    last_line = line
                else:
                    current_paragraph.append(line)
                    last_line = line
                    

    if current_paragraph:
        paragraph_info = extract_paragraph_info(current_paragraph, indent_factor=indent_factor)
        paragraphs.append(paragraph_info)

    return paragraphs

@log_time
@progress_monitor
def process_ltpages(doc: List[LTPage], char_margin_factor: float = 4.0, line_margin_factor: float = 0.5, margins: Optional[Tuple[float]] = None, bbox_overlap: float = 1.0, verbose: bool = False) -> List[PageInfo]:
    pages = []
    for page_num, page in enumerate(doc, start=1):
        clip = (margins[0], margins[1], page.width - margins[2], page.height - margins[3]) if margins else None

        paragraphs = extract_paragraphs_from_page(page, pagenum=page_num, char_margin_factor=char_margin_factor, line_margin_factor=line_margin_factor, clip=clip, bbox_overlap=bbox_overlap)
        if paragraphs is None:
            pages.append(PageInfo(text="", bbox=page.bbox, paragraphs=[], font_sizes=[], char_widths=[], colors=set(), starts_with_indent=False, split_end_paragraph=False))
            continue
        page_info = extract_page_info(paragraphs)
        page_info.update_pagenum(page_num, recursive=True)
        pages.append(page_info)
    return pages

def main():
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Process PDF and extract text information.")
        parser.add_argument("pdf_path", type=str, help="Path to the PDF file to be processed.")
        parser.add_argument("-o", "--out", type=str, default="processed_pages.pkl", help="Output file path for the processed pages.")
        parser.add_argument("-sd", "--save_doc", action="store_true", help="Save the document object [list of LTPages] to a file.")
        parser.add_argument("-d", "--doc_path", type=str, help="Path to the document object [list of LTPages] file.")
        parser.add_argument("-gd", "--generate_doc", action="store_true", help="Process the PDF from scratch.")
        parser.add_argument("-m", "--margins", type=float, nargs=4, default=(0, 20, 0, 15), help="Margins for clipping the page (left, top, right, bottom).")
        parser.add_argument("-bbo", "--bbox_overlap", type=float, default=0.9, help="Bounding box overlap factor.")
        parser.add_argument("-cm", "--char_margin", type=float, default=4, help="Character margin factor.")
        parser.add_argument("-lm", "--line_margin", type=float, default=0.6, help="Line margin factor.")
        parser.add_argument("-lo", "--line_overlap", type=float, default=0.6, help="Line overlap factor.")
        parser.add_argument("-v", "--verbose", action="store_true", help="Display progress bar.")
        parser.add_argument("--debug", action="store_true", help="Set logging level to config.DEBUG.")
        return parser.parse_args()

    args = parse_arguments()
    config.set_debug(args.debug, args.verbose)
    
    params = LAParams(line_overlap=args.line_overlap, char_margin=args.char_margin, line_margin=args.line_margin)
    

    if config.DEBUG:
        max_pages = 10
        pdf_path = "/home/rookslog/pdf2anki/examples/pathmarks_ocr.pdf"
        doc_path = os.path.splitext(pdf_path)[0] + f"_doc_test.pkl"
        print(f"Debug mode is on. Processing only the first {max_pages} pages.")
    else:
        max_pages = 0
        pdf_path = "/home/rookslog/pdf2anki/examples/pathmarks_ocr.pdf" if args.__dict__.get("pdf_path", None) is None else args.pdf_path
        doc_path = os.path.splitext(pdf_path)[0] + "_doc.pkl" if args.__dict__.get("doc_path", None) is None else args.doc_path

    
    if os.path.exists(doc_path) and not args.generate_doc:

        print(f"Document exists at {doc_path}. Loading...")
        try:
            start_time = time.time()    
            doc = load_and_restore_pages(doc_path)
            end_time = time.time()
            print("Loaded in {:.4f} seconds.\n".format(end_time - start_time))
        except IOError as e:
            logging.error(f"Error loading document: {e}")
            print("Processing PDF...")
            start_time = time.time()
            doc = tuple(extract_pages(pdf_path, laparams=params, maxpages=max_pages))
            end_time = time.time()
            print(f"Processed {len(doc)} pages in {end_time - start_time:.4f} seconds.\n")
        except EOFError as e:
            logging.error(f"Error loading document: {e} \nDocument is empty.\n")
            print("Deleting document and processing PDF...")
            os.remove(doc_path)
            start_time = time.time()
            doc = tuple(extract_pages(pdf_path, laparams=params, maxpages=max_pages))
            end_time = time.time()
            print(f"Processed {len(doc)} pages in {end_time - start_time:.4f} seconds.\n")
    else:
        if args.generate_doc:
            print(f"generate_doc flag is set. Processing PDF...")
        else:
            print(f"Document does not exist at {doc_path}. Processing PDF...")
        start_time = time.time()
        logging.info(f"extract_pages_args: {pdf_path}, {params}, {max_pages}")
        doc = tuple(extract_pages(pdf_path, laparams=params, maxpages=max_pages))
        end_time = time.time()
        print(f"Processed {len(doc)} pages in {end_time - start_time:.4f} seconds.\n")
              
    if args.save_doc:
        print(f"Saving document to {doc_path}...")
        try:
            start_time = time.time()
            save_pages_file_objs(doc, doc_path)
            end_time = time.time()
            print(f"Succesfully saved document in {end_time - start_time:.4f} seconds.\n")
        except Exception as e:
            logging.error(f"Error saving document: {e}")
            print("Deleting empty file...")
            os.remove(doc_path)
            print("Deleted. Continuing...")

    margins = args.margins
    bbox_overlap = args.bbox_overlap

    processed_pages = process_ltpages(doc, char_margin_factor=params.char_margin, line_margin_factor=params.line_margin, margins=margins, bbox_overlap=bbox_overlap, verbose=args.verbose)
    print(f"Processed {len(processed_pages)} pages.")

    output_file_path = os.splitext(args.pdf_path)[0] + "processed_pages.pkl" if args.__dict__.get("out", None) is None else args.out
    print(f"Saving processed pages to {output_file_path}...")
    with open(output_file_path, "wb") as output_file:
        pickle.dump(processed_pages, output_file)
    print("Done.")

if __name__ == "__main__":
    main()
