import os
from utils import log_time
from typing import List, Optional, Tuple, Pattern, Union
from itertools import chain
from pdfminer.layout import LTPage, LTTextBoxHorizontal, LTTextLineHorizontal, LTFigure, LTChar, LAParams
from pdfminer.high_level import extract_pages
from pdfminer.pdfparser import PDFDocument
from extraction import extract_lines_from_figure
import re
from filters import ToCFilter
from pdf2anki.extraction import PageInfo, ParagraphInfo, LineInfo, CharInfo
from pdf2anki.filters import ToCFilterVars, ToCFilterOptions, ToCEntry
from dataclasses import dataclass
from typing import Optional, List, Dict, Iterator, Tuple
from itertools import chain
from collections import defaultdict



class FoundGreedy(Exception):
    """A hacky solution to do short-circuiting in Python.

    The main reason to do this short-circuiting is to untangle the logic of
    greedy filter with normal execution, which makes the typing and code much
    cleaner, but it can also save some unecessary comparisons.

    Probably similar to call/cc in scheme or longjump in C
    c.f. https://ds26gte.github.io/tyscheme/index-Z-H-15.html#node_sec_13.2
    """
    level: int

    def __init__(self, level):
        """
        Argument
          level: level of the greedy filter
        """
        super().__init__()
        self.level = level


def blk_to_str(blk: dict) -> str:
    """Extract all the text inside a block"""
    return " ".join([
        spn.get('text', "").strip()
        for line in blk.get('lines', [])
        for spn in line.get('spans', [])
    ])

def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2 ** 0:
        l.append("superscript")
    if flags & 2 ** 1:
        l.append("italic")
    if flags & 2 ** 2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2 ** 3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2 ** 4:
        l.append("bold")
    return ", ".join(l)

@dataclass
class Fragment:
    """A fragment of the extracted heading"""
    text: str
    level: int
    bbox: tuple

def concat_bboxes(bboxes):
    """
    Combine a list of bounding boxes into a single bounding box.

    Args:
        bboxes (list of tuples): List of bounding boxes, where each bounding box is represented as a tuple (x0, y0, x1, y1).

    Returns:
        tuple: A tuple representing the combined bounding box (min_x0, min_y0, max_x1, max_y1).
    """
    if not bboxes:
        return None

    min_x0 = min(bbox[0] for bbox in bboxes)
    min_y0 = min(bbox[1] for bbox in bboxes)
    max_x1 = max(bbox[2] for bbox in bboxes)
    max_y1 = max(bbox[3] for bbox in bboxes)

    return (min_x0, min_y0, max_x1, max_y1)


def concatFrag(frags: Iterator[Optional[Fragment]], sep: str = " ") -> Dict[int, Tuple[str, tuple]]:
    """Concatenate fragments to strings

    Returns
      a dictionary (level -> (title, bbox)) that contains the title and bbox for each level.
    """
    # accumulate a list of strings and bboxes for each level of heading
    acc = defaultdict(lambda: ([], []))
    for frag in frags:
        if frag is not None:
            acc[frag.level][0].append(frag.text)
            acc[frag.level][1].append(frag.bbox)

    result = {}
    for level, (strs, bboxes) in acc.items():
        result[level] = (sep.join(strs), concat_bboxes(bboxes))
    return result


class Recipe:
    """The internal representation of a recipe using dataclasses."""
    filters: List[ToCFilter]

    def __init__(self, recipe_list: List[List[ToCFilterVars, ToCFilterOptions]]):
        fltr_dicts = recipe_list.get('heading', [])

        if not fltr_dicts:
            raise ValueError("no filters found in recipe")
        self.filters = [ToCFilter(fltr) for fltr in fltr_dicts]

    @classmethod
    def from_dict_list(cls, data: List[Dict]):
        """
        Initialize Recipe from a list of dictionaries.
        """
        recipe_dict = {"heading": data}
        return cls(recipe_dict)

    @classmethod
    def from_nested_list(cls, data: List[List[Union[ToCFilterVars, ToCFilterOptions]]]):
        """
        Initialize Recipe from a nested list structure.
        """
        recipe_dict = {"heading": []}
        for sublist in data:
            for item in sublist:
                filter_dict = {
                    "level": item.var,
                    "font": item.options.font,
                    "size": item.options.size,
                    "color": item.options.color,
                    "char_width": item.options.char_width,
                    "is_upper": item.options.is_upper,
                    "bbox": item.options.bbox,
                    "greedy": item.options.greedy
                }
                recipe_dict["heading"].append(filter_dict)
        return cls(recipe_dict)

    def extract_meta(self, pages: List[PageInfo]):
        """
        Extract meta information from a list of PageInfo objects.
        """
        for page in pages:
            meta = self.search_in_page(page)
            # Process meta as needed

    def generate_recipe(self, pages: List[PageInfo]):
        """
        Generate a recipe from a list of PageInfo objects.
        """
        self.extract_meta(pages)
        # Additional logic to generate recipe

    def search_in_page(self, page: PageInfo) -> List[Fragment]:
        """
        Search a single PageInfo object for relevant data.
        """
        fragments = []
        for paragraph in page.paragraphs:
            for line in paragraph.lines:
                fragments.extend(self._extract_line(line))
        return fragments

    def extract_page(self, page: PageInfo) -> List[ToCEntry]:
        """
        Iteratively call extract_paragraph for each paragraph in a PageInfo and return list of ToCEntries.
        """
        toc_entries = []
        for paragraph in page.paragraphs:
            entry = self.extract_paragraph(paragraph, page.page_number)
            if entry:
                toc_entries.append(entry)
        return toc_entries

    def extract_paragraph(self, paragraph: ParagraphInfo, pagenum: int) -> Optional[ToCEntry]:
        """
        Extract a paragraph into a ToCEntry using ParagraphInfo dataclass.
        """
        try:
            frags = chain.from_iterable([
                self._extract_line(line) for line in paragraph.lines
            ])
            titles = concatFrag(frags)
            return ToCEntry(
                level=titles['level'],
                title=titles['title'],
                page_range=pagenum,
                vpos=paragraph.vpos,
                bbox=paragraph.bbox,
                text=paragraph.text,
                subsections=[]
            )
        except FoundGreedy as e:
            # Handle greedy extraction
            return ToCEntry(
                level=0,
                title=paragraph.text,
                page_range=pagenum,
                vpos=paragraph.vpos,
                bbox=paragraph.bbox
            )


@log_time
def search_in_page(regex: re.Pattern, 
                   page_num: int, 
                   page: PageInfo, 
                   ign_pattern: Optional[Pattern] = None, 
                   char_margin_factor: float = 0.5, 
                   clip: Optional[Tuple[float]] = None) -> List[dict]:
    """Search for `text` in `page` and extract meta

    Arguments
      regex: the compiled regex pattern to search for
      page_num: the page number (1-based index)
      text_info: the extracted text information from the PDF
      ign_pattern: an optional pattern to ignore certain text

    Returns
      a list of meta
    """
    def clean_text(text):
        return ' '.join(text.split()).replace('“', '"').replace('”', '"').replace("’", "'").lstrip()

    result = []
    page_lines = extract_lines_from_figure(page, char_margin_factor=char_margin_factor, clip=clip)

    for line in page_lines:
        line_text = clean_text(line["text"])
        if regex.search(line_text):
            is_upper = re.sub(ign_pattern, "", line_text).isupper() if ign_pattern else line_text.isupper()
            result.append({
                "page_num": page_num,
                "text": line_text,
                "font": line["font"],
                "color": line["color"],
                "size": line["size"],
                "char_width": line["char_width"],
                "bbox": line["bbox"],
                "is_upper": is_upper
            })

    return result

@log_time
def extract_meta(doc: List[PageInfo], 
                 pattern: str, 
                 page_numbers: Optional[List[int]] = None,
                 ign_case: bool = False, 
                 ign_pattern: Optional[Pattern] = None,
                 char_margin_factor: float = 0.5,
                 clip: Optional[Tuple[float]] = None) -> List[dict]:
    meta = []
    regex = re.compile(pattern, re.IGNORECASE) if ign_case else re.compile(pattern)

    if page_numbers is None:
        pages = enumerate(doc, start=1)
    else:
        for i, pagenum in enumerate(page_numbers):
            pages = [(pagenum+1, doc[i])]

    for pagenum, page in pages:
        meta.extend(search_in_page(regex, pagenum, page, ign_pattern=ign_pattern, char_margin_factor=char_margin_factor, clip=clip))
    
    return meta


def generate_recipe(doc: List[PageInfo], 
                    headers: List[Tuple[str, int, int]], 
                    page_numbers: List[int] = None,
                    tolerances: dict = {"font": 1e-1, "bbox": 1e-1}, 
                    ign_pattern=None, 
                    char_margin_factor: float = 0.5, 
                    clip: Optional[Tuple[float]] = None) -> Recipe:
    recipe_dict = {"heading": []}

    for header, level, pagenum in headers:
        # Extract metadata for the header
        # Really shitty code here sorry
        page_index = page_numbers.index(pagenum-1) if page_numbers else pagenum - 1
        page = doc[page_index]
        meta = extract_meta([page], header,
                            [page_numbers[page_index]] if page_numbers is not None else None, 
                            ign_case=True, ign_pattern=ign_pattern, 
                            char_margin_factor=char_margin_factor, clip=clip)
        if len(meta) > 1:
            meta.sort(key=lambda x: x["char_width"])
        line = meta[-1]
        font_filter = {
            "name": line["font"],
            "size": line["size"], 
            "size_tolerance": tolerances["font"],
            "color": line["color"],
            "char_width": line["char_width"],
            "is_upper": line["is_upper"]
        }
        if level == 1:
            bbox_filter = {
                "left": line["bbox"][0],
                "top": line["bbox"][1],
                "right": line["bbox"][2],
                "bottom": line["bbox"][3],
                "tolerance": tolerances["bbox"]
            }
        else:
            bbox_filter = {}

        recipe_dict["heading"].append({
            "level": level,
            "font": font_filter,
            "bbox": bbox_filter,
            "greedy": False
        })
    recipe = Recipe(recipe_dict)
    return recipe

@log_time
def extract_toc(doc: List[PageInfo], recipe: Recipe, page_range=None):
    toc_entries = []
    if page_range is None:
        page_range = [0, len(doc) - 1]

    for page_num in range(page_range[0]-1, page_range[1]):
        page = doc[page_num]
        for element in page:
            if isinstance(element, (LTTextBoxHorizontal, LTTextLineHorizontal, LTFigure)):
                toc_entries.extend(recipe.extract_paragraph(element, page_num))

    toc_entries = merge_toc_entries(toc_entries)
    return toc_entries

@log_time
def merge_toc_entries(toc_entries, tolerance=10):
    merged_entries = []
    toc_entries.sort(key=lambda x: (x.page, x.level, x.vpos))  # Sort by page, level, and vertical position

    i = 0
    while i < len(toc_entries):
        current_entry = toc_entries[i]
        j = i + 1
        while j < len(toc_entries) and toc_entries[j].level == current_entry.level and toc_entries[j].page == current_entry.page:
            if abs(toc_entries[j].vpos - current_entry.vpos) <= tolerance:
                # Merge titles
                current_entry.title += " " + toc_entries[j].title
                j += 1
            else:
                break
        merged_entries.append(current_entry)
        i = j

    return merged_entries

@log_time
def nest_toc_entries(flat_toc):
    if not flat_toc:
        return []

    def nest_entries(entries, current_level):
        nested = []
        while entries:
            entry = entries[0]
            if entry.level > current_level:
                if nested:
                    nested[-1].subsections = nest_entries(entries, entry.level)
            elif entry.level < current_level:
                break
            else:
                nested.append(entries.pop(0))
        return nested

    return nest_entries(flat_toc, flat_toc[0].level)

def main():   
    pdf_path = os.get(__file__) + "/examples/pathmarks_ocr.pdf"
    params = LAParams(line_margin=0.5, char_margin=2.0, line_overlap=0.5)
    page_range = [16, 60]
    text_doc = list(extract_pages(pdf_path, laparams=params, maxpages=page_range[1]))
    # recipe = generate_recipe(text_doc, [("Table of Contents", 1, 16)], page_numbers=range(16, 60))
    # toc_entries = extract_toc(text_doc, recipe, page_range=page_range)
    # merged_toc_entries = merge_toc_entries(toc_entries, tolerance=30)

if __name__ == "__main__":
    main()
