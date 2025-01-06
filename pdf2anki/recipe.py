from copy import copy
from enum import Enum
import hashlib
import json
import logging
import os
import pickle
from pdf2anki.extraction import merge_paragraphs, merge_split_paragraphs
from pdf2anki.utils import clean_text, contained_in_bbox, get_text_index_from_vpos
from pdf2anki.decorators import log_time
from typing import Callable, List, Literal, Optional, Set, Tuple, Pattern, Union, overload
from pdfminer.layout import LAParams
from pdfminer.high_level import extract_pages
import re
from pdf2anki.elements import PageInfo, ParagraphInfo, LineInfo, ElementType
from pdf2anki.filters import ToCFilterOptions, ToCEntry, ToCFilter, TextFilterOptions, TextFilter, FontFilterOptions, BoundingBoxFilterOptions, average_float_vars, find_similar_filter_sets, merge_similar_text_filters
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
                        title=paragraph.text.strip() if not toc_filter.opts.greedy else paragraph.text.strip(),
                        pagenum=pagenum,
                        page_range=[pagenum, pagenum],
                        start_vpos=paragraph.bbox[1],  # bottom y-value as vertical position
                        bbox=paragraph.bbox,
                        text="",
                        subsections=[]
                    )
                    # Associate matched TextFilters for later use
                    entry.text_filter_ids = self.toc_to_text_map.get(toc_filter_id, {})
                    toc_entries.append(entry)
                    break  # if multiple filters match, we can stop at the first
        return toc_entries

    def extract_text_for_headers(self, doc: List[PageInfo], toc_entries: List[ToCEntry], page_range: Optional[Tuple[int, int]] = None) -> 'Recipe':
        # TODO: Change the logic for this such that the end_page is actually the last page of the current entry and not the first page of the next entry
        # This would require some check to see if the end_vpos as it currently stands is near the top of the page and to use the end_vpos of the final paragraph
        # from the previous page
        # Will also need to change the logic of nest_toc_entries
        if page_range:
            assert len(page_range) == 2 and page_range[0] <= page_range[1]
        else:
            page_range = (1, len(doc))
            
        for i, entry in enumerate(toc_entries):
            filtered_pages = []
            end_page = toc_entries[i+1].pagenum if i+1 < len(toc_entries) else page_range[1]
            entry.page_range[1] = end_page
            end_vpos = toc_entries[i+1].bbox[3] if i+1 < len(toc_entries) else 0
            entry.end_vpos = end_vpos
            
            # paragraphs = [paragraph for paragraph in doc[entry.page_range[0]].paragraphs if paragraph.bbox[3] <= entry.start_vpos]\
            #              + [paragraph for page in doc[entry.page_range[0]+1:entry.page_range[1]-1] for paragraph in page.paragraphs]\
            #              + [paragraph for paragraph in doc[entry.page_range[1]-1].paragraphs if paragraph.bbox[3] > end_vpos]
            # merged_paragraphs = merge_split_paragraphs(paragraphs)
            previous_page_split = False
            pages = doc[entry.page_range[0]-1:end_page]
            for j, page in enumerate(pages):
                filtered_page = []
                paragraphs = [paragraph for paragraph in page.paragraphs]
                if page.pagenum == entry.page_range[0]: # or if j == 0
                    paragraphs = [paragraph for paragraph in paragraphs if paragraph.bbox[3] <= entry.start_vpos]
                if page.pagenum == end_page: # or if j == len(pages)-1
                    paragraphs = [paragraph for paragraph in paragraphs if paragraph.bbox[3] > end_vpos]
                for k, paragraph in enumerate(paragraphs):
                    for text_filter_id in entry.text_filter_ids:
                        text_filter = self.text_filters[text_filter_id]
                        if k == 0 and previous_page_split:
                            merged_paragraph = merge_paragraphs([pages[j-1].paragraphs[-1], paragraph])
                            previous_page_split = False
                        elif k == len(paragraphs)-1 and paragraph.split_end_line:
                            merged_paragraph = merge_paragraphs([paragraph, pages[j+1].paragraphs[0]])
                            previous_page_split = True
                        else:
                            merged_paragraph = paragraph
                        if text_filter.admits(merged_paragraph):
                            filtered_page.append(merged_paragraph.text)
                            break
                filtered_pages.append('\n'.join(filtered_page))

            entry.text = "\n\n".join(filtered_pages)
        return self

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
        merged_all_toc_entries = merge_toc_entries(all_toc_entries, tolerance=30)
        if extract_text:
            self.extract_text_for_headers(doc, merged_all_toc_entries, page_range)

        return all_toc_entries
    
def merge_similar_text_filters_in_recipe(recipe: Recipe, tolerance_dict: Dict[str, float]) -> Recipe:
    """
    Merge similar TextFilters in a Recipe instance while maintaining the proper mapping structure.

    Args:
        recipe (Recipe): The Recipe instance.
        tolerance_dict (Dict[str, float]): The tolerance dictionary with keys "bbox" and "font".

    Returns:
        Recipe: The updated Recipe instance with merged TextFilters.
    """
    # Find sets of similar text filters
    text_filters_list = list(recipe.text_filters.values())
    similar_sets = find_similar_filter_sets(text_filters_list, tolerance_dict)

    # Merge similar text filters
    merged_filters = [average_float_vars(text_filters_list, indices) for indices in similar_sets]

    # Create a mapping from old text filter IDs to new text filter IDs
    old_to_new_id_map = {}
    for indices, merged_filter in zip(similar_sets, merged_filters):
        new_id = recipe._generate_filter_id(merged_filter)
        for index in indices:
            old_id = recipe._generate_filter_id(text_filters_list[index])
            old_to_new_id_map[old_id] = new_id

    # Update the text filters in the recipe
    recipe.text_filters = {recipe._generate_filter_id(merged_filter): merged_filter for merged_filter in merged_filters}

    # Update the toc_to_text_map in the recipe
    new_toc_to_text_map = {}
    for toc_filter_id, text_filter_ids in recipe.toc_to_text_map.items():
        new_text_filter_ids = {old_to_new_id_map[old_id] for old_id in text_filter_ids}
        new_toc_to_text_map[toc_filter_id] = new_text_filter_ids

    recipe.toc_to_text_map = new_toc_to_text_map

    return recipe

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
                  start_vpos: Optional[float] = None, 
                  ign_pattern: Optional[Pattern] = None,
                  clip: Optional[Tuple[float]] = None, 
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
                       start_vpos: Optional[float] = None, 
                       ign_pattern: Optional[Pattern] = None,
                       clip: Optional[Tuple[float]] = None, 
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
                 start_vpos: Optional[float] = None, 
                 ign_pattern: Optional[Pattern] = None,
                 clip: Optional[Tuple[float]] = None, 
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
    if start_vpos is None:
        start_vpos = clip[1] if clip is not None else page.bbox[3]
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
                   start_vpos: Optional[float] = None, 
                   ign_pattern: Optional[Pattern] = None,
                   clip: Optional[Tuple[float]] = None, 
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
    return extract_function(regex, page, start_vpos, clip=clip, ign_pattern=ign_pattern, tolerance=tolerance)

@log_time
def extract_elements(doc: List[PageInfo], 
                 pattern: str, 
                 page_numbers: Optional[List[int]] = None,
                 ign_case: bool = True, 
                 ign_pattern: Optional[Pattern] = None,
                 tolerance: float = DEFAULT_TOLERANCE,
                 clip: Optional[Tuple[float]] = None,
                 element_type: ElementType = ElementType.LINE) -> List[Union[LineInfo, ParagraphInfo, PageInfo]]:
    all_elements = []
    regex = re.compile(pattern, re.IGNORECASE) if ign_case else re.compile(pattern)

    if page_numbers is None:
        pages = enumerate(doc, start=1)
    else:
        pages = [(pagenum, doc[pagenum-1]) for pagenum in page_numbers]

    for pagenum, page in pages:
        elements = [element for element in search_in_page(regex, page, ign_pattern=ign_pattern, clip=clip, tolerance=tolerance, element_type=element_type) if isinstance(element, Union[LineInfo, ParagraphInfo, PageInfo])]
        [element.update_pagenum(pagenum, recursive=True) for element in elements]
        all_elements.extend(elements)
    return all_elements

@log_time
def generate_recipe(doc: List[PageInfo], 
                    headers: List[Dict[str, Union[Tuple[int, str], List[Tuple[int, str]], int]]], 
                    tolerances: dict = {"font": 1e-1, "bbox": 1e-1}, 
                    ign_pattern=None, 
                    ign_case=True,
                    clip: Optional[Tuple[float]] = None,
                    include_text_filters: bool = False,
                    merge_similar_text_filters: bool = True,
                    toc_filter_options: Optional[List[Dict[str, Union[ToCFilterOptions, FontFilterOptions, BoundingBoxFilterOptions]]]] = None,
                    text_filter_options: Optional[List[Dict[str, Union[TextFilterOptions, FontFilterOptions, BoundingBoxFilterOptions]]]] = None) -> Recipe:
    header_filters: List[ToCFilter] = []
    all_text_filters: List[List[TextFilter]] = []


    if toc_filter_options is None:
        toc_filter_options = [{} for _ in headers]
    
    if text_filter_options is None:
        text_filter_options = [{} for _ in headers]

    for i, header in enumerate(headers):
        header_text, header_level = header["header"], header["level"]
        header_page_num, header_str = header_text

        # Extract metadata for the header
        # page_index = page_numbers.index(header_page_num - 1) if page_numbers else header_page_num - 1
        # page = doc[page_index]
        header_elements = [header_element for header_element in \
                            extract_elements(doc, header_str, [header_page_num], ign_case=ign_case, \
                                                ign_pattern=ign_pattern, tolerance=tolerances["bbox"], clip=clip, element_type=ElementType.PARAGRAPH)\
                                if isinstance(header_element, ParagraphInfo)]

        if len(header_elements) > 1:
            header_elements.sort(key=lambda x: x.char_width)
        paragraph = header_elements[-1] if len(header_elements) > 0 else None

        if paragraph is None:
            logging.warning(f"No header found for {header_str} on page {header_page_num}.")
            continue
        
        fltr_dict = {
            "toc": toc_filter_options[i].get("toc", ToCFilterOptions()),
            "font": toc_filter_options[i].get("font", FontFilterOptions()),
            "bbox": toc_filter_options[i].get("bbox", BoundingBoxFilterOptions())
        }

        toc_filter = ToCFilter.from_paragraph_info(paragraph, header_level, fltr_dict)
        header_filters.append(toc_filter)

        if include_text_filters:
            text_filters: List[TextFilter] = []
            for (text_page_num, text_str) in header["text"]:
                # page_index = page_numbers.index(text_page_num - 1) if page_numbers else text_page_num - 1
                # page = doc[page_index] 

                text_elements = set([text_element for text_element in \
                                 extract_elements(doc, text_str, [text_page_num],
                                                    ign_pattern=ign_pattern, tolerance=tolerances["bbox"], clip=clip, element_type=ElementType.PARAGRAPH) \
                                        if isinstance(text_element, ParagraphInfo)])
                
                if len(text_elements) > 1:
                    logging.warning(f"Multiple text elements found for header {header_str} on page {text_page_num}. Using the first element.")

                for text_element in text_elements:
                    assert isinstance(text_element, ParagraphInfo)
                    opts = {"text": text_filter_options[i].get("text", TextFilterOptions()),
                            "font": text_filter_options[i].get("font", FontFilterOptions()),
                            "bbox": text_filter_options[i].get("bbox", BoundingBoxFilterOptions())}
                    text_filter = TextFilter.from_paragraph_info(text_element, opts)
                    text_filters.append(text_filter)
            all_text_filters.append(text_filters)
    
    
    recipe_dict = {
        "heading": header_filters,
        "text": all_text_filters if include_text_filters else []
    }
    recipe = Recipe(recipe_dict)
    return recipe if not merge_similar_text_filters else merge_similar_text_filters_in_recipe(recipe, tolerances)

@log_time
def merge_toc_entries(toc_entries, tolerance=30):
    merged_entries = []
    toc_entries.sort(key=lambda x: (x.pagenum, x.level, -x.start_vpos))  # Sort by page, level, and vertical position

    i = 0
    while i < len(toc_entries):
        current_entry = toc_entries[i]
        j = i + 1
        while j < len(toc_entries) and toc_entries[j].level == current_entry.level and toc_entries[j].pagenum == current_entry.pagenum:
            if abs(toc_entries[j].start_vpos - current_entry.start_vpos) <= tolerance:
                # Merge titles
                current_entry.title += " " + toc_entries[j].title
                j += 1
            else:
                break
        merged_entries.append(current_entry)
        i = j

    return merged_entries

@log_time
def nest_toc_entries(flat_toc: List[ToCEntry]) -> List[ToCEntry]:
    if not flat_toc:
        return []

    def nest_entries(entries, current_level):
        nested = []
        while entries:
            entry = entries[0]
            if entry.level > current_level: # it is a subsection of the current entry
                if nested:
                    nested[-1].subsections = nest_entries(entries, entry.level)
                    nested[-1].page_range[1] = nested[-1].subsections[-1].page_range[1]
            elif entry.level < current_level: # it is a parent entry
                break
            else: # it is a sibling entry, entry_level == current_level
                nested.append(entries.pop(0))
                if len(nested) > 1:
                    nested[-2].page_range[1] = nested[-1].page_range[0]
                    nested[-2].end_vpos = nested[-1].bbox[3]
        return nested

    return nest_entries(flat_toc, flat_toc[0].level)

def toc_to_dict(toc_entries: List[ToCEntry], include_metadata: bool = False) -> List[ToCEntry]:
   for toc_entry in toc_entries:
         toc_entry_dict = {
              "level": toc_entry.level,
              "title": toc_entry.title,
              "pagenum": toc_entry.pagenum,
              "page_range": toc_entry.page_range,
              "start_vpos": toc_entry.start_vpos,
              "end_vpos": toc_entry.end_vpos,
              "bbox": toc_entry.bbox,
              "text": toc_entry.text,
              "subsections": toc_to_dict(toc_entry.subsections, include_metadata)
         }
         if include_metadata:
              toc_entry_dict["text_filter_ids"] = toc_entry.text_filter_ids
         yield toc_entry_dict
        


def save_toc_entries(toc_entries: List[ToCEntry], output_path):
    with open(output_path, "w") as f:
        json.dump([entry.to_() for entry in toc_entries], f, indent=4)

def main():   
    def parse_args():
        parser = argparse.ArgumentParser(description="Generate a recipe from a PDF document.")
        parser.add_argument("--pdf_path", type=str, help="Path to the PDF file.")
        parser.add_argument("--processed_pages_path", type=str, default=None, help="Path to the processed pages pickle file.")
        parser.add_argument("--recipe_config", type=str, default=None, help="Path to the JSON file containing headers and options.")
        parser.add_argument("--page_numbers", type=str, default=None, help="Comma-separated list of page numbers to include.")
        parser.add_argument("--tolerances", type=str, default='{"font": 1e-1, "bbox": 1e-1}', help="JSON string of tolerances for font and bbox.")
        parser.add_argument("--ign_pattern", type=str, default=None, help="Regex pattern to ignore.")
        parser.add_argument("--clip", type=str, default=None, help="Bounding box to clip the search area.")
        parser.add_argument("--include_text_filters", action='store_true', help="Include text filters in the recipe.")
        return parser.parse_args()

    args = parse_args()
    pdf_path = "/home/rookslog/pdf2anki/examples/pathmarks_ocr.pdf"
    processed_pages_path = os.path.splitext(pdf_path)[0] + "_processed_pages.pkl"
    with open(processed_pages_path, "rb") as f:
        doc = pickle.load(f)

    headers = [
        {"header": (16, "Comments on Karl Faspers"), "level": 1, "text": [(16, "A \"fitting\" orientation for a positive and illuminating critical review of")]},
        {"header": (54, "Phenomenology and Theology"), "level": 1, "text": [(55, "The popular understanding of the relationship between theology and")]},
        {"header": (58, "THE POSITIVE CHARACTER OF THEOLOGY"), "level": 2, "text": [(58, "A positive science is the founding disclosure of a being that is given")]},
        {"header": (112, "ON THE ESSENCE OF GROUND"), "level": 1, "text": [(113, "but was concerned with understanding their interconnection")]},
        {"header": (122, "TRANSCENDENCE AS THE DOMAIN OF THE"), "level": 2, "text": [(122, "A preliminary remark on terminology must guide our use of")]},
        {"header": (165, "TRUTH AS ERRANCY"), "level": 2, "text": [(165, "As insistent, the human being is turned toward the most readily available")]},
        {"header": (140, "III. ON THE ESSENCE OF GROUND*"), "level": 2, "text": [(140, "reason or ground to the domain of transcendence")]},    ]
    ign_pattern = re.compile(r'^[a-z]\)')

    tolerances = {"font": 3e-1, "bbox": 1e-1, "text": 4e-1}

    toc_filter_options: list[dict] = []
    text_filter_options: list[dict] = []

    for header in headers:
        toc_filter_option = {
            "toc": ToCFilterOptions(check_bbox=False), 
            "font": FontFilterOptions(check_colors=False, check_width=header["level"]==2, check_is_upper=header["level"] == 2, size_tolerance=tolerances["font"], ign_pattern=ign_pattern),
            "bbox": BoundingBoxFilterOptions()}
        toc_filter_options.append(toc_filter_option)

        if args.include_text_filters or True:
            text_filter_option = {
                "text": TextFilterOptions(check_bbox=False, tolerance=tolerances["text"]),
                "font": FontFilterOptions(check_colors=False, check_is_upper=False, size_tolerance=tolerances["font"]),
                "bbox": None}
            text_filter_options.append(text_filter_option)

    
    recipe = generate_recipe(doc, headers, 
                             tolerances={"font": 3e-1, "bbox": 3e-1}, 
                             ign_pattern=ign_pattern, clip=None, 
                             include_text_filters=True,
                             toc_filter_options=toc_filter_options,
                             text_filter_options=text_filter_options)
    assert isinstance(recipe, Recipe)
    page_range = [16,len(doc)-22]
    toc_entries = recipe.extract_toc(doc, extract_text=True, page_range=page_range)
    merged_toc_entries = merge_toc_entries(toc_entries, tolerance=30)
    nested_toc_entries = nest_toc_entries(merged_toc_entries)

    

    print(recipe)
    print(nested_toc_entries)

if __name__ == "__main__":
    main()
