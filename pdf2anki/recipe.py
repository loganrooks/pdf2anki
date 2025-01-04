from enum import Enum
import hashlib
import json
import os
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
    
    def _generate_filter_id(filter_obj) -> str:
        """Generate a SHA256 hash for a filter object based on its attributes."""
        filter_json = json.dumps(filter_obj.__dict__, sort_keys=True)
        return hashlib.sha256(filter_json.encode('utf-8')).hexdigest()

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
