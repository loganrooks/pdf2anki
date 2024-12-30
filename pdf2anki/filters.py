"""Filter on span dictionaries

This module contains the internal representation of heading filters, which are
used to test if a span should be included in the ToC.
"""

import re

from typing import Optional
from re import Pattern

DEF_TOLERANCE: dict = {"font": 1e-1, "bbox": 1e-1}


def admits_float(expect: Optional[float],
                 actual: Optional[float],
                 tolerance: float) -> bool:
    """Check if a float should be admitted by a filter"""
    return (expect is None) or \
           (actual is not None and abs(expect - actual) <= tolerance)


class FontFilter:
    '''
    Filter on font attributes.
    Attributes:
        name (Pattern): Compiled regex pattern for font name.
        avg_size (Optional[float]): Average font size.
        max_size (Optional[float]): Maximum font size.
        size_tolerance (float): Tolerance for font size comparison.
        color (Optional[int]): Font color.
        is_upper (Optional[bool]): Whether the text is uppercase.
        flags (int): Bitwise flags for font attributes (superscript, italic, serif, monospace, bold).
        ign_mask (int): Bitwise mask for ignored font attributes.
        ign_pattern (Optional[Pattern]): Compiled regex pattern for ignored text.
    Methods:
        __init__(font_dict: dict):
            Initializes the FontFilter with the given font attributes.
        admits(line: dict) -> bool:
            Checks if the font attributes admit the span.
            Args:
                line (dict): The line dictionary to be checked.
            Returns:
                bool: False if the span doesn't match current font attribute, True otherwise.
    '''


    name: Pattern
    size: Optional[float]
    size_tolerance: Optional[float]
    color: Optional[int]
    width: Optional[float]
    is_upper: Optional[bool]
    flags: Optional[int]
    ign_pattern: Optional[Pattern]
    # besides the usual true (1) and false (0), we have another state,
    # unset (x), where the truth table would be
    # a b diff?
    # 0 0 0
    # 0 1 1
    # 1 0 1
    # 1 1 0
    # x 0 0
    # x 1 0
    # it's very inefficient to compare bit by bit, which would take 5 bitwise
    # operations to compare, and then 4 to combine the results, we will use a
    # trick to reduce it to 2 ops.
    # step 1: use XOR to find different bits. if unset, set bit to 0, we will
    #         take care of false positives in the next step
    # a b a^b
    # 0 0 0
    # 0 1 1
    # 1 0 1
    # 1 1 0
    # step 2: use AND with a ignore mask, (0 for ignored) to eliminate false
    #         positives
    # a b a&b
    # 0 1 0           <- no diff
    # 0 0 0           <- no diff
    # 1 1 1           <- found difference
    # 1 0 0           <- ignored
    ign_mask: int

    def __init__(self, font_dict: dict):
        self.name = re.compile(font_dict.get('name', ""))
        self.size = font_dict.get('size')
        self.width = font_dict.get('char_width')
        self.size_tolerance = font_dict.get('size_tolerance', DEF_TOLERANCE['font'])
        self.color = font_dict.get('color')
        self.is_upper = font_dict.get('is_upper')
        # self.line_height = font_dict.get('line_height')
        # some branchless trick, mainly to save space
        # x * True = x
        # x * False = 0
        self.flags = (0b00001 * font_dict.get('superscript', False) |
                      0b00010 * font_dict.get('italic', False) |
                      0b00100 * font_dict.get('serif', False) |
                      0b01000 * font_dict.get('monospace', False) |
                      0b10000 * font_dict.get('bold', False))

        self.ign_mask = (0b00001 * ('superscript' in font_dict) |
                         0b00010 * ('italic' in font_dict) |
                         0b00100 * ('serif' in font_dict) |
                         0b01000 * ('monospace' in font_dict) |
                         0b10000 * ('bold' in font_dict))
        self.ign_pattern = font_dict.get('ign_pattern', None)

    def __repr__(self):
        return f"FontFilter({self.name}, size: {self.size}, size_tolerance: {self.size_tolerance}, width: {self.width}, color: {self.color}, is_upper: {self.is_upper}, flags: {self.flags}, ign_mask: {self.ign_mask}, ign_pattern: {self.ign_pattern})"
    def admits_line(self, line: dict) -> bool:
        """Check if the font attributes admit the span

        Argument
          spn: the span dict to be checked
        Returns
          False if the span doesn't match current font attribute
        """
        fonts = [span.get("font", "") for span in line.get("spans", {})]
        

        filtered_spans = [span for span in line.get("spans", {}) if self.ign_pattern is None or not self.ign_pattern.search(span.get("text", ""))]
        line_text = "".join([span.get("text", "") for span in filtered_spans])
        
        if len(line_text) < 5:
            return False

        if not self.name.search(" ".join(fonts)):
            return False

        if self.color is not None and not self.color in line["color"]:
            return False

        if not admits_float(self.size, line["size"], self.size_tolerance):
            return False
        
        if not admits_float(self.width, line["width"], self.size_tolerance):
            return False
        
        if not self.is_upper is None:
            return self.is_upper == line_text.isupper()

        flags = line.get('flags', ~self.flags)
        # see above for explanation
        return not (flags ^ self.flags) & self.ign_mask


class BoundingBoxFilter:
    """Filter on bounding boxes"""
    left: Optional[float]
    top: Optional[float]
    right: Optional[float]
    bottom: Optional[float]
    tolerance: float

    def __init__(self, bbox_dict: dict):
        self.left = bbox_dict.get('left')
        self.top = bbox_dict.get('top')
        self.right = bbox_dict.get('right')
        self.bottom = bbox_dict.get('bottom')
        self.tolerance = bbox_dict.get('tolerance', DEF_TOLERANCE['bbox'])

    def admits(self, line: dict) -> bool:
        """Check if the bounding box admit the span

        Argument
          spn: the span dict to be checked
        Returns
          False if the span doesn't match current bounding box setting
        """
        bbox = line.get('bbox', (None, None, None, None))
        return (admits_float(self.left, bbox[0], self.tolerance) and
                admits_float(self.top, bbox[1], self.tolerance) and
                admits_float(self.right, bbox[2], self.tolerance) and
                admits_float(self.bottom, bbox[3], self.tolerance))
    


class ToCFilter:
    """Filter on span dictionary to pick out headings in the ToC"""
    # The level of the title, strictly > 0
    level: int
    # When set, the filter will be more *greedy* and extract all the text in a
    # block even when at least one match occurs
    greedy: bool
    font: FontFilter
    bbox: BoundingBoxFilter

    def __init__(self, fltr_dict: dict):
        lvl = fltr_dict.get('level')

        if lvl is None:
            raise ValueError("filter's 'level' is not set")
        if lvl < 1:
            raise ValueError("filter's 'level' must be >= 1")

        self.level = lvl
        self.greedy = fltr_dict.get('greedy', False)
        self.font = FontFilter(fltr_dict.get('font', {}))
        self.bbox = BoundingBoxFilter(fltr_dict.get('bbox', {}))

    def admits(self, line: dict) -> bool:
        """Check if the filter admits the span

        Arguments
          spn: the span dict to be checked
        Returns
          False if the span doesn't match the filter
        """
        return self.font.admits_line(line) and self.bbox.admits(line)