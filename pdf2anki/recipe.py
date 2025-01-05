from copy import copy
from enum import Enum
import hashlib
import json
import logging
import os
import pickle
from pdf2anki.utils import clean_text, contained_in_bbox, get_text_index_from_vpos
from pdf2anki.decorators import log_time
from typing import Callable, List, Literal, Optional, Set, Tuple, Pattern, Union, overload
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_pages
import re
from pdf2anki.elements import PageInfo, ParagraphInfo, LineInfo, ElementType
from pdf2anki.filters import ToCFilterOptions, ToCEntry, ToCFilter, TextFilterOptions, TextFilter, FontFilterOptions, BoundingBoxFilterOptions
from dataclasses import dataclass
from typing import Optional, List, Dict, Iterator, Tuple
from collections import defaultdict
from pdf2anki.config import DEBUG
import argparse

DEFAULT_TOLERANCE = 1e-2

class SearchMode(Enum):
    HEADER = 'header'
    TEXT = 'text'

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
    toc_filters: Dict[str, ToCFilter]
    text_filters: Dict[str, TextFilter]
    toc_to_text_map: Dict[str, Set[str]]

    def __init__(self, filters_dict: Dict[str, Union[List[ToCFilter], List[List[TextFilter]]]]):
        toc_filters = filters_dict.get('heading', [])
        text_filters_list = filters_dict.get('text', [])

        # Generate unique IDs for ToCFilters and TextFilters
        self.toc_filters = {self._generate_filter_id(toc_filter): toc_filter for toc_filter in toc_filters}
        all_text_filters = [text_filter for sublist in text_filters_list for text_filter in sublist]
        unique_text_filters = {self._generate_filter_id(text_filter): text_filter for text_filter in all_text_filters}
        self.text_filters = unique_text_filters

        # Create the mapping from ToCFilter IDs to TextFilter IDs
        self.toc_to_text_map = {}
        for toc_filter, text_filters in zip(toc_filters, text_filters_list):
            toc_filter_id = self._generate_filter_id(toc_filter)
            text_filter_ids = set([self._generate_filter_id(text_filter) for text_filter in text_filters])
            self.toc_to_text_map[toc_filter_id] = text_filter_ids

    def __repr__(self):
        return f"Recipe(toc_filters={self.toc_filters},\n\t text_filters={self.text_filters},\n\t toc_to_text_map={self.toc_to_text_map})"

    @staticmethod
    def generate_filter_id(filter_obj) -> str:
        """Generate a SHA256 hash for a filter object based on its attributes."""
        filter_json = json.dumps(filter_obj.__dict__, sort_keys=True)
        return hashlib.sha256(filter_json.encode('utf-8')).hexdigest()

    @classmethod
    def from_dict(cls, recipe_dict: Dict[str, Union[List[Dict], List[List[Dict]]]]) -> 'Recipe':
        toc_dicts = recipe_dict.get('heading', [])
        text_dicts_list = recipe_dict.get('text', [])

        toc_filters = [ToCFilter.from_dict(fltr) for fltr in toc_dicts]
        text_filters_list = [[TextFilter.from_dict(fltr) for fltr in text_dicts] for text_dicts in text_dicts_list]

        return cls({
            'heading': toc_filters,
            'text': text_filters_list
        })

    @classmethod
    def from_nested_dict(cls, nested_dict: Dict[str, Dict[str, Union[List[Dict], List[List[Dict]]]]]) -> 'Recipe':
        toc_dicts = nested_dict.get('heading', {}).get('filters', [])
        text_dicts_list = nested_dict.get('text', {}).get('filters', [])
        
        toc_filters = [ToCFilter.from_dict(fltr) for fltr in toc_dicts]
        text_filters_list = [[TextFilter.from_dict(fltr) for fltr in text_dicts] for text_dicts in text_dicts_list]

        return cls({
            'heading': toc_filters,
            'text': text_filters_list
        })

    @classmethod
    def from_lists(cls, toc_list: List[ToCFilter], text_lists: List[List[TextFilter]]) -> 'Recipe':
        return cls({
            'heading': toc_list,
            'text': text_lists
        })

    def get_text_filters_for_toc(self, toc_filter: ToCFilter) -> Optional[List[TextFilter]]:
        toc_filter_id = self.generate_filter_id(toc_filter)
        text_filter_ids = self.toc_to_text_map.get(toc_filter_id, [])
        return [self.text_filters[text_filter_id] for text_filter_id in text_filter_ids]
    
    def _generate_filter_id(self, filter_obj) -> str:
        """Generate a SHA256 hash for a filter object based on its attributes."""
    # Generate a SHA256 hash from the JSON string
        return hash(filter_obj)

    def add_toc_filter(self, toc_filter: ToCFilter, text_filters: Optional[List[TextFilter]] = None):
        """
        Add a ToCFilter to the Recipe and optionally associate it with TextFilters.
        """
        toc_filter_id = self._generate_filter_id(toc_filter)
        self.toc_filters[toc_filter_id] = toc_filter

        if text_filters:
            text_filter_ids = []
            for text_filter in text_filters:
                text_filter_id = self._generate_filter_id(text_filter)
                self.text_filters[text_filter_id] = text_filter
                text_filter_ids.append(text_filter_id)
            self.toc_to_text_map[toc_filter_id] = text_filter_ids

    def remove_toc_filter(self, toc_filter: ToCFilter):
        """
        Remove a ToCFilter from the Recipe.
        """
        toc_filter_id = self._generate_filter_id(toc_filter)
        if toc_filter_id in self.toc_filters:
            del self.toc_filters[toc_filter_id]
            if toc_filter_id in self.toc_to_text_map:
                del self.toc_to_text_map[toc_filter_id]

    def add_text_filter(self, toc_filter: ToCFilter, text_filter: TextFilter):
        """
        Add a TextFilter to the Recipe and associate it with a ToCFilter.
        """
        toc_filter_id = self._generate_filter_id(toc_filter)
        if toc_filter_id not in self.toc_filters:
            raise ValueError(f"Referenced ToCFilter {toc_filter} not found in ToCFilters")

        text_filter_id = self._generate_filter_id(text_filter)
        self.text_filters[text_filter_id] = text_filter

        if toc_filter_id in self.toc_to_text_map:
            self.toc_to_text_map[toc_filter_id].add(text_filter_id)
        else:
            self.toc_to_text_map[toc_filter_id] = [text_filter_id]

    def remove_text_filter(self, text_filter: TextFilter):
        """
        Remove a TextFilter from the Recipe.
        """
        text_filter_id = self._generate_filter_id(text_filter)
        if text_filter_id in self.text_filters:
            del self.text_filters[text_filter_id]

            # Remove the text filter from all ToCFilter associations
            for toc_filter_id, text_filter_ids in self.toc_to_text_map.items():
                if text_filter_id in text_filter_ids:
                    text_filter_ids.remove(text_filter_id)
                    if not text_filter_ids:
                        del self.toc_to_text_map[toc_filter_id]

    def extract_headers(self, paragraphs: List[ParagraphInfo], pagenum: int) -> List[ToCEntry]:
        toc_entries = []
        for paragraph in paragraphs:
            for toc_filter_id, toc_filter in self.toc_filters.items():
                if toc_filter.admits(paragraph):
                    entry = ToCEntry(
                        level=toc_filter.vars.level,
                        title=paragraph.text.strip(),
                        page=pagenum,
                        vpos=paragraph.bbox[3],  # top y-value as vertical position
                        bbox=paragraph.bbox,
                        text=paragraph.text,
                        subsections=[]
                    )
                    # Associate matched TextFilters for later use
                    entry.text_filter_ids = self.toc_to_text_map.get(toc_filter_id, {})
                    toc_entries.append(entry)
                    break  # if multiple filters match, we can stop at the first
        return toc_entries

    def extract_text_for_headers(self, doc: List[PageInfo], toc_entries: List[ToCEntry]) -> None:
        for i, entry in enumerate(toc_entries):
            merged_text = []
            end_page = toc_entries[i+1].pagenum if i+1 < len(toc_entries) else None
            end_vpos = toc_entries[i+1].bbox[3] if i+1 < len(toc_entries) else None
            for page in doc:
                if page.pagenum < entry.pagenum:
                    continue
                if end_page is not None and page.pagenum >= end_page:
                    break

                for paragraph in page.paragraphs:
                    if page.pagenum == entry.pagenum and paragraph.bbox[3] > entry.start_vpos:
                        # skip paragraphs above the header
                        continue
                    if end_page is not None and page.pagenum == end_page - 1 and paragraph.bbox[3] <= end_vpos:
                        # stop at next header's vpos
                        break
                    for text_filter_id in entry.text_filter_ids:
                        text_filter = self.text_filters[text_filter_id]
                        if text_filter.admits(paragraph):
                            merged_text.append(paragraph.text)
                            break

            entry.text = entry.title + "\n\n" + "\n".join(merged_text)

    def extract_toc(self, doc: List[PageInfo], page_range: Optional[Tuple[int, int]] = None, extract_text: bool = False) -> List[ToCEntry]:
        all_toc_entries = []
        if page_range:
            start_page, end_page = page_range
            if start_page < 1 or end_page > len(doc) or start_page > end_page:
                raise ValueError("Invalid page range specified.")
        else:
            start_page, end_page = 1, len(doc)

        for page in doc[start_page-1:end_page]:
            page_entries = self.extract_headers(page.paragraphs, page.pagenum)
            all_toc_entries.extend(page_entries)

        if extract_text:
            self.extract_text_for_headers(doc, all_toc_entries)

        return all_toc_entries

@overload
def search_in_page(regex: re.Pattern, 
                   page: PageInfo, 
                   start_vpos: Optional[float], 
                   ign_pattern: Optional[Pattern],
                   clip: Optional[Tuple[float]], 
                   tolerance: float, 
                   element_type: Literal[ElementType.LINE]) -> List[LineInfo]: ...

@overload
def search_in_page(regex: re.Pattern, 
                   page: PageInfo, 
                   start_vpos: Optional[float], 
                   ign_pattern: Optional[Pattern],
                   clip: Optional[Tuple[float]], 
                   tolerance: float, 
                   element_type: Literal[ElementType.PARAGRAPH]) -> List[ParagraphInfo]: ...

@overload
def search_in_page(regex: re.Pattern, 
                   page: PageInfo, 
                   start_vpos: Optional[float], 
                   ign_pattern: Optional[Pattern],
                   clip: Optional[Tuple[float]], 
                   tolerance: float, 
                   element_type: Literal[ElementType.PAGE]) -> List[PageInfo]: ...

def extract_lines(regex: re.Pattern, 
                  page: PageInfo, 
                  start_vpos: Optional[float], 
                  ign_pattern: Optional[Pattern],
                  clip: Optional[Tuple[float]], 
                  tolerance: float = DEFAULT_TOLERANCE) -> List[LineInfo]:
    result = []
    page_lines = [line for paragraph in page.paragraphs for line in paragraph.lines if contained_in_bbox(line.bbox, clip, bbox_overlap=1-tolerance)] \
                        if clip is not None else [line for paragraph in page.paragraphs for line in paragraph.lines]
    
    start_index = 0
    if start_vpos is not None:
        vpos = page.bbox[3]
        while vpos > start_vpos:
            start_index += 1
            vpos = page_lines[start_index].bbox[3]

    for line in page_lines[start_index:]:
        line_text = clean_text(line.text)
        if regex.search(line_text):
            result.append(line)
    return result

def extract_paragraphs(regex: re.Pattern, 
                       page: PageInfo, 
                       start_vpos: Optional[float], 
                       ign_pattern: Optional[Pattern],
                       clip: Optional[Tuple[float]], 
                       tolerance: float = DEFAULT_TOLERANCE) -> List[ParagraphInfo]:
    """
    Extract paragraphs from a page that match a given regex pattern.
    
    Args:
        regex: Regular expression pattern to search for.
        page: PageInfo object representing the page.
        start_vpos: Vertical position to start searching from.
        clip: Bounding box to clip the search area.
        tolerance: Tolerance for bbox overlap.
        ign_pattern: Pattern to ignore.
        
    Returns:
        List[ParagraphInfo]: List of paragraphs that match the regex pattern.
    """
    result = []
    paragraphs = [paragraph for paragraph in page.paragraphs if contained_in_bbox(paragraph.bbox, clip, bbox_overlap=1-tolerance)] \
                    if clip is not None else page.paragraphs
    
    start_index = 0
    if start_vpos is not None:
        vpos = page.bbox[3]
        while vpos > start_vpos:
            start_index += 1
            vpos = paragraphs[start_index].bbox[3]
    
    for paragraph in paragraphs:
        paragraph_text = clean_text(paragraph.text)
        if regex.search(paragraph_text):
            result.append(paragraph)
    return result


def extract_page(regex: re.Pattern, 
                 page: PageInfo, 
                 start_vpos: Optional[float], 
                 ign_pattern: Optional[Pattern],
                 clip: Optional[Tuple[float]], 
                 tolerance: float = DEFAULT_TOLERANCE) -> Optional[PageInfo]:
    """
    If regex pattern in page, return page.

    Args:
        regex: Regular expression pattern to search for.
        page: PageInfo object representing the page.
        start_vpos: Vertical position to start searching from.
        tolerance: Tolerance for bbox overlap.
        clip: Bounding box to clip the search area.

    Returns:
        PageInfo: The page if the pattern is found, otherwise None.
    """
    start_index = get_text_index_from_vpos(start_vpos, page)
    page_text = clean_text(page.text[start_index:])
    if regex.search(page_text):
        return page
    
# Dispatcher dictionary
extract_dispatcher: Dict[ElementType, Callable[..., Union[List[LineInfo], List[ParagraphInfo], PageInfo]]] = {
    ElementType.LINE: extract_lines,
    ElementType.PARAGRAPH: extract_paragraphs,
    ElementType.PAGE: extract_page
}

@log_time
def search_in_page(regex: re.Pattern, 
                   page: PageInfo, 
                   start_vpos: Optional[float], 
                   ign_pattern: Optional[Pattern],
                   clip: Optional[Tuple[float]], 
                   tolerance: float = DEFAULT_TOLERANCE,
                   element_type: ElementType = ElementType.LINE) -> Union[List[LineInfo], List[ParagraphInfo], PageInfo]:

    """
    Search for a regex pattern in a page and return the matching element.

    Args:
        regex (re.Pattern): The compiled regex pattern to search for.
        page_num (int): The page number.
        page (PageInfo): The page to search in.
        element_type (ElementType): The type of element to search for.
        start_vpos (Optional[float]): The vertical position to start searching from.
        char_margin_factor (float): The character margin factor.
        clip (Optional[Tuple[float]]): The clipping box.
        ign_pattern (Optional[Pattern]): The pattern to ignore.

    Returns:
        Union[List[LineInfo], List[ParagraphInfo], List[LTFigure]]: A list of matching elements.
    """
    extract_function = extract_dispatcher[element_type]
    return extract_function(regex, page, start_vpos, clip, ign_pattern)

@log_time
def extract_elements(doc: List[PageInfo], 
                 pattern: str, 
                 page_numbers: Optional[List[int]] = None,
                 ign_case: bool = False, 
                 ign_pattern: Optional[Pattern] = None,
                 tolerance: float = DEFAULT_TOLERANCE,
                 clip: Optional[Tuple[float]] = None,
                 element_type: ElementType = ElementType.LINE) -> List[Union[LineInfo, ParagraphInfo, PageInfo]]:
    all_elements = []
    regex = re.compile(pattern, re.IGNORECASE) if ign_case else re.compile(pattern)

    if page_numbers is None:
        pages = enumerate(doc, start=1)
    else:
        pages = [(pagenum, doc[pagenum - 1]) for pagenum in page_numbers]

    for pagenum, page in pages:
        elements = [element for element in search_in_page(regex, pagenum, page, ign_pattern=ign_pattern, clip=clip, tolerance=tolerance, element_type=element_type) if isinstance(element, Union[LineInfo, ParagraphInfo, PageInfo])]
        [element.update_pagenum(pagenum, recursive=True) for element in elements]
        all_elements.extend(elements)
    return all_elements
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
