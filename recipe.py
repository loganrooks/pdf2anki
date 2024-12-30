
import os
from utils import log_time
from typing import List, Optional
from itertools import chain
from pdfminer.layout import LTPage, LTTextBoxHorizontal, LTTextLineHorizontal, LTFigure, LTChar, LAParams
from pdfminer.high_level import extract_pages




class ToCEntry:
    def __init__(self, level, title, page, vpos):
        self.level = level
        self.title = title
        self.page = page
        self.vpos = vpos  # Vertical position
        self.text = ""
        self.subsections = []

    def __repr__(self):
        return f"ToCEntry(level={self.level}, title={self.title}, page={self.page}, vpos={self.vpos}, text={self.text})"

class Recipe:
    """The internal representation of a recipe"""
    filters: List[ToCFilter]

    def __init__(self, recipe_dict: dict):
        fltr_dicts = recipe_dict.get('heading', [])

        if len(fltr_dicts) == 0:
            raise ValueError("no filters found in recipe")
        self.filters = [ToCFilter(fltr) for fltr in fltr_dicts]

  

    def _extract_line(self, line: dict) -> List[Optional[Fragment]]:
        """Extract matching heading fragments in a line.

        Argument
          line: a line dictionary
          {
            'bbox': (float, float, float, float),
            'wmode': int,
            'dir': (float, float),
            'spans': [dict]
          }
        Returns
          a list of fragments concatenated from result in a line
        """
        for fltr in self.filters:
            if fltr.admits(line):
                text = ''.join([span['text'] for span in line.get('spans', {})])
                bbox = line.get('bbox', (0, 0, 0, 0))

                if not text:
                    # don't match empty spaces
                    return None

                if fltr.greedy:
                    # propagate all the way back to extract_block
                    raise FoundGreedy(fltr.level)

                return [Fragment(char['text'], fltr.level, bbox) for char in line.get('chars', {})]
        return [None for _ in range(len(line.get("chars", {})))]

    def extract_block(self, block: dict, page: int) -> List[ToCEntry]:
        """Extract matching headings in a block.

        Argument
          block: a block dictionary
          {
            'bbox': (float, float, float, float),
            'lines': [dict],
            'type': int
          }
        Returns
          a list of toc entries, concatenated from the result of lines
        """
        if block.get('type') != 0:
            # not a text block
            return []
        
        bbox = block.get('bbox', (0, 0, 0, 0))
        vpos = bbox[1]

        try:
            frags = chain.from_iterable([
                self._extract_line(ln) for ln in block.get('lines')
            ])
            titles = concatFrag(frags)
            return [
                ToCEntry(level, title, page, bbox[1], bbox)
                for level, (title, bbox) in titles.items()
            ]
        except FoundGreedy as e:
            # return the entire block as a single entry
            print(f"FoundGreedy exception: {e}")
            return [ToCEntry(0, block.get('text', ''), page, vpos, bbox)]

@log_time
def extract_toc(doc: List[LTPage], recipe: Recipe, page_range=None):
    toc_entries = []
    if page_range is None:
        page_range = [0, len(doc) - 1]

    for page_num in range(page_range[0]-1, page_range[1]):
        page = doc[page_num]
        for element in page:
            if isinstance(element, (LTTextBoxHorizontal, LTTextLineHorizontal, LTFigure)):
                toc_entries.extend(recipe.extract_block(element, page_num))

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
    page_range = [16, 60]
    text_doc = list(extract_pages(pdf_path, laparams=params, maxpages=page_range[1]))
    toc_entries = extract_toc(text_doc, recipe, page_range=page_range)
    merged_toc_entries = merge_toc_entries(toc_entries, tolerance=30)

if __name__ == "__main__":
    main()
