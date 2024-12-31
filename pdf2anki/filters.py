"""Filter on span dictionaries

This module contains the internal representation of heading filters, which are
used to test if a span should be included in the ToC.
"""

from re import Pattern

import re
from typing import Dict, Optional, Tuple, Union, List, override
from extraction import ParagraphInfo, LineInfo
from dataclasses import dataclass
from collections import Counter

from pdf2anki.utils import get_average, concat_bboxes

DEF_TOLERANCE: dict = {"font": 1e-1, "bbox": 1e-1}



@dataclass
class FontFilterOptions:
    """Options for configuring the FontFilter."""
    check_name: bool = True
    check_color: bool = True
    check_size: bool = True
    check_width: bool = True
    check_is_upper: bool = False
    size_tolerance: float = DEF_TOLERANCE["font"]
    ign_mask: int = 0
    ign_pattern: Optional[Pattern] = None

@dataclass
class FontFilterVars:
    """Variables for the FontFilter."""
    name: Optional[str] = None
    color: Optional[str] = None
    font_size: Optional[float] = None
    char_width: Optional[float] = None
    is_upper: Optional[bool] = None
   

@dataclass
class BoundingBoxFilterOptions:
    """Options for configuring the BoundingBoxFilter."""
    check_left: bool = True
    check_top: bool = True
    check_right: bool = True
    check_bottom: bool = True
    tolerance: float = DEF_TOLERANCE["bbox"]

@dataclass
class BoundingBoxFilterVars:
    """Variables for the BoundingBoxFilter."""
    left: Optional[float] = None
    top: Optional[float] = None
    right: Optional[float] = None
    bottom: Optional[float] = None

def admits_float(expect: Optional[float], 
                 actual: Optional[float], 
                 tolerance: float) -> bool:
    """
    Check if a float should be admitted by a filter.

    Args:
        expect (Optional[float]): The expected value.
        actual (Optional[float]): The actual value.
        tolerance (float): The tolerance for comparison.

    Returns:
        bool: True if the actual value is within the tolerance of the expected value, False otherwise.
    """
    return (expect is None) or (actual is not None and abs(expect - actual) <= tolerance)


class FontFilter:
    def __init__(self, vars: FontFilterVars, opts: FontFilterOptions):
        """
        Initialize a FontFilter.

        Args:
            vars (FontFilterVars): The variables for the filter.
            opts (FontFilterOptions): The options for configuring the filter.
        """
        self.vars = vars
        self.opts = opts

    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[FontFilterOptions] = None) -> 'FontFilter':
        """
        Create a FontFilter from a ParagraphInfo object.

        Args:
            paragraph_info (ParagraphInfo): The ParagraphInfo object.
            opts (Optional[FontFilterOptions]): The options for configuring the filter.

        Returns:
            FontFilter: The created FontFilter.
        """
        if opts is None:
            opts = FontFilterOptions()
        vars = FontFilterVars()
        if paragraph_info:
            vars.name = '//'.join(paragraph_info.fonts)
            vars.color = '//'.join(paragraph_info.colors)
            vars.font_size = paragraph_info.font_size
            vars.char_width = paragraph_info.char_width
            vars.is_upper = paragraph_info.text.isupper() if opts.ign_pattern is None else re.sub(opts.ign_pattern, "", paragraph_info.text).isupper()
        return cls(vars, opts)

    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[FontFilterOptions] = None) -> 'FontFilter':
        """
        Create a FontFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            opts (Optional[FontFilterOptions]): The options for configuring the filter.

        Returns:
            FontFilter: The created FontFilter.
        """
        if opts is None:
            opts = FontFilterOptions()
        vars = FontFilterVars(
            name=line_info.font,
            color=line_info.color,
            font_size=line_info.size,
            char_width=line_info.char_width,
            is_upper=line_info.text.isupper() if opts.ign_pattern is None else re.sub(opts.ign_pattern, "", line_info.text).isupper()
        )
        return cls(vars, opts)
    
    @override
    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[Dict[str, Union[bool, float]]] = None) -> 'FontFilter':
        """
        Create a FontFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            opts (Optional[Dict[str, Union[bool, float]]]): The options for configuring the filter.

        Returns:
            FontFilter: The created FontFilter.
        """
        if opts is None:
            opts = FontFilterOptions()
        else:
            opts = FontFilterOptions(**opts)
        vars = FontFilterVars(
            name=line_info.font,
            color=line_info.color,
            font_size=line_info.size,
            char_width=line_info.char_width,
            is_upper=line_info.text.isupper() if opts.ign_pattern is None else re.sub(opts.ign_pattern, "", line_info.text).isupper()
        )
        return cls(vars, opts)

    @classmethod
    def from_line_info_list(cls, line_info_list: List[LineInfo], opts: Optional[FontFilterOptions] = None) -> 'FontFilter':
        """
        Create a FontFilter from a list of LineInfo objects.

        Args:
            line_info_list (List[LineInfo]): The list of LineInfo objects.
            opts (Optional[FontFilterOptions]): The options for configuring the filter.

        Returns:
            FontFilter: The created FontFilter.
        """
        if opts is None:
            opts = FontFilterOptions()
        vars = FontFilterVars(
            name="//".join([line_info.font for line_info in line_info_list]),
            color="//".join([line_info.color for line_info in line_info_list]),
            font_size=get_average([line_info.size for line_info in line_info_list]),
            char_width=get_average([line_info.char_width for line_info in line_info_list]),
            is_upper=all([line_info.text.isupper() for line_info in line_info_list]) if opts.ign_pattern is None 
                        else all([re.sub(opts.ign_pattern, "", line_info.text).isupper() for line_info in line_info_list])
        )
        return cls(vars, opts)
    
    @classmethod
    def from_dict(cls, fltr_dict: Dict[str, Union[str, float, bool]]) -> 'FontFilter':
        """
        Create a FontFilter from a dictionary.

        Args:
            fltr_dict (Dict): The dictionary containing filter configuration.

        Returns:
            FontFilter: The created FontFilter.
        """
        opts = FontFilterOptions(
            check_name=fltr_dict.get('check_name', True),
            check_color=fltr_dict.get('check_color', True),
            check_size=fltr_dict.get('check_size', True),
            check_width=fltr_dict.get('check_width', True),
            check_is_upper=fltr_dict.get('check_is_upper', False),
            size_tolerance=fltr_dict.get('size_tolerance', DEF_TOLERANCE["font"]),
            ign_pattern=fltr_dict.get('ign_pattern')
        )
        vars = FontFilterVars(
            name=fltr_dict.get('name'),
            color=fltr_dict.get('color'),
            font_size=fltr_dict.get('font_size'),
            char_width=fltr_dict.get('char_width'),
            is_upper=fltr_dict.get('is_upper')
        )
        return cls(vars, opts)

    def admits(self, line_info: LineInfo) -> bool:
        """
        Check if a LineInfo object is admitted by the filter.

        Args:
            line_info (LineInfo): The LineInfo object.

        Returns:
            bool: True if the LineInfo object is admitted, False otherwise.
        """
        if self.opts.check_name and not self.vars.name in line_info.font:
            return False
        if self.opts.check_color and not self.vars.color in line_info.color:
            return False
        if self.opts.check_size and not admits_float(self.vars.font_size, line_info.size, self.opts.size_tolerance):
            return False
        if self.opts.check_width and not admits_float(self.vars.char_width, line_info.char_width, self.opts.size_tolerance):
            return False
        if self.opts.check_is_upper and (self.vars.is_upper != line_info.text.isupper() if self.opts.ign_pattern is None \
                else re.sub(self.opts.ign_pattern, "", line_info.text).isupper()):
            return False
        return True

    def __repr__(self):
        return (f"FontFilter(vars={self.vars}, opts={self.opts})")


class BoundingBoxFilter:
    def __init__(self, vars: BoundingBoxFilterVars, opts: BoundingBoxFilterOptions):
        """
        Initialize a BoundingBoxFilter.

        Args:
            vars (BoundingBoxFilterVars): The variables for the filter.
            opts (BoundingBoxFilterOptions): The options for configuring the filter.
        """
        self.vars = vars
        self.opts = opts

    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[BoundingBoxFilterOptions] = None) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a ParagraphInfo object.

        Args:
            paragraph_info (ParagraphInfo): The ParagraphInfo object.
            opts (Optional[BoundingBoxFilterOptions]): The options for configuring the filter.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """
        if opts is None:
            opts = BoundingBoxFilterOptions()
        vars = BoundingBoxFilterVars(
            left=paragraph_info.bbox[0],
            top=paragraph_info.bbox[1],
            right=paragraph_info.bbox[2],
            bottom=paragraph_info.bbox[3]
        )
        return cls(vars, opts)
    
    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[BoundingBoxFilterOptions] = None) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            opts (Optional[BoundingBoxFilterOptions]): The options for configuring the filter.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """
        if opts is None:
            opts = BoundingBoxFilterOptions()
        vars = BoundingBoxFilterVars(
            left=line_info.bbox[0],
            top=line_info.bbox[1],
            right=line_info.bbox[2],
            bottom=line_info.bbox[3]
        )
        return cls(vars, opts)
    
    @override
    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[Dict[str, Union[bool, float]]] = None) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            opts (Optional[Dict[str, Union[bool, float]]]): The options for configuring the filter.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """

        if opts is None:
            opts = BoundingBoxFilterOptions()
        else:
            opts = BoundingBoxFilterOptions(**opts)

        vars = BoundingBoxFilterVars(
            left=line_info.bbox[0],
            top=line_info.bbox[1],
            right=line_info.bbox[2],
            bottom=line_info.bbox[3]
        )
        return cls(vars, opts)
    
    @classmethod
    def from_line_info_list(cls, line_info_list: List[LineInfo], opts: Optional[BoundingBoxFilterOptions] = None) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a list of LineInfo objects.

        Args:
            line_info_list (List[LineInfo]): The list of LineInfo objects.
            opts (Optional[BoundingBoxFilterOptions]): The options for configuring the filter.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """
        if opts is None:
            opts = BoundingBoxFilterOptions()
        lines_bbox = concat_bboxes([line_info.bbox for line_info in line_info_list])
        vars = BoundingBoxFilterVars(
            left=lines_bbox[0],
            top=lines_bbox[1],
            right=lines_bbox[2],
            bottom=lines_bbox[3]
        )
        return cls(vars, opts)
    
    @classmethod
    def from_dict(cls, fltr_dict: Dict[str, float]) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a dictionary.

        Args:
            fltr_dict (Dict): The dictionary containing filter configuration.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """
        opts = BoundingBoxFilterOptions(
            check_left=fltr_dict.get('check_left', True),
            check_top=fltr_dict.get('check_top', True),
            check_right=fltr_dict.get('check_right', True),
            check_bottom=fltr_dict.get('check_bottom', True),
            tolerance=fltr_dict.get('tolerance', DEF_TOLERANCE["bbox"])
        )
        vars = BoundingBoxFilterVars(
            left=fltr_dict.get('left'),
            top=fltr_dict.get('top'),
            right=fltr_dict.get('right'),
            bottom=fltr_dict.get('bottom')
        )
        return cls(vars, opts)

    def admits(self, bbox: Tuple[float, float, float, float]) -> bool:
        """
        Check if a bounding box is admitted by the filter.

        Args:
            bbox (Tuple[float, float, float, float]): The bounding box.

        Returns:
            bool: True if the bounding box is admitted, False otherwise.
        """
        if self.opts.check_left and not admits_float(self.vars.left, bbox[0], self.opts.tolerance):
            return False
        if self.opts.check_top and not admits_float(self.vars.top, bbox[1], self.opts.tolerance):
            return False
        if self.opts.check_right and not admits_float(self.vars.right, bbox[2], self.opts.tolerance):
            return False
        if self.opts.check_bottom and not admits_float(self.vars.bottom, bbox[3], self.opts.tolerance):
            return False
        return True

    def __repr__(self):
        return (f"BoundingBoxFilter(vars={self.vars}, opts={self.opts})")

    
@dataclass
class ToCFilterOptions:
    """Options for configuring the ToCFilter."""
    check_font: bool = True
    check_bbox: bool = True
    greedy: bool = False

@dataclass
class ToCFilterVars:
    """Variables for the ToCFilter."""
    level: int
    font: FontFilter
    bbox: BoundingBoxFilter

class ToCFilter:
    def __init__(self, vars: ToCFilterVars, opts: ToCFilterOptions):
        """
        Initialize a ToCFilter.

        Args:
            vars (ToCFilterVars): The variables for the filter.
            opts (ToCFilterOptions): The options for configuring the filter.
        """
        self.vars = vars
        self.opts = opts

    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, 
                            fltr_dict: Dict[str, Union[bool, FontFilterOptions, BoundingBoxFilterOptions]]) -> 'ToCFilter':
        """
        Create a ToCFilter from a ParagraphInfo object.

        Args:
            paragraph_info (ParagraphInfo): The ParagraphInfo object.
            fltr_dict (Dict): The dictionary containing filter configuration.

        Returns:
            ToCFilter: The created ToCFilter.
        """
        lvl = fltr_dict.get('level')
        if lvl is None:
            raise ValueError("filter's 'level' is not set")
        if lvl < 1:
            raise ValueError("filter's 'level' must be >= 1")

        opts = ToCFilterOptions(
            check_font=fltr_dict.get('check_font', True),
            check_bbox=fltr_dict.get('check_bbox', True),
            greedy=fltr_dict.get('greedy', False)
        )

        font_filter = FontFilter.from_paragraph_info(paragraph_info, fltr_dict.get('font', {}))
        bbox_filter = BoundingBoxFilter.from_paragraph_info(paragraph_info, fltr_dict.get('bbox', {}))

        vars = ToCFilterVars(
            level=lvl,
            font=font_filter,
            bbox=bbox_filter
        )

        return cls(vars, opts)
    
    @classmethod
    def from_line_info(cls, line_info: LineInfo, 
                       fltr_dict: Dict[str, Union[int, bool, FontFilterOptions, BoundingBoxFilterOptions]]) -> 'ToCFilter':
        """
        Create a ToCFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            fltr_dict (Dict): The dictionary containing filter configuration.

        Returns:
            ToCFilter: The created ToCFilter.
        """
        lvl = fltr_dict.get('level')
        if lvl is None:
            raise ValueError("filter's 'level' is not set")
        if lvl < 1:
            raise ValueError("filter's 'level' must be >= 1")

        opts = ToCFilterOptions(
            check_font=fltr_dict.get('check_font', True),
            check_bbox=fltr_dict.get('check_bbox', True),
            greedy=fltr_dict.get('greedy', False)
        )

        font_filter = FontFilter.from_line_info(line_info, fltr_dict.get('font', {}))
        bbox_filter = BoundingBoxFilter.from_line_info(line_info, fltr_dict.get('bbox', {}))

        vars = ToCFilterVars(
            level=lvl,
            font=font_filter,
            bbox=bbox_filter
        )

        return cls(vars, opts)
    
    @override
    @classmethod
    def from_line_info(cls, line_info: LineInfo,
                       fltr_dict: Dict[str, Union[int, bool, Dict[str, Union[str, float, bool]]]]) -> 'ToCFilter':
        """
        Create a ToCFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            fltr_dict (Dict): The dictionary containing filter configuration.

        Returns:
            ToCFilter: The created ToCFilter.
        """
        lvl = fltr_dict.get('level')
        if lvl is None:
            raise ValueError("filter's 'level' is not set")
        if lvl < 1:
            raise ValueError("filter's 'level' must be >= 1")
        
        opts = ToCFilterOptions(
            check_font=fltr_dict.get('check_font', True),
            check_bbox=fltr_dict.get('check_bbox', True),
            greedy=fltr_dict.get('greedy', False)
        )

        font_filter = FontFilter.from_line_info(line_info, fltr_dict.get('font', {}))
        bbox_filter = BoundingBoxFilter.from_line_info(line_info, fltr_dict.get('bbox', {}))

        vars = ToCFilterVars(
            level=lvl,
            font=font_filter,
            bbox=bbox_filter
        )

        return cls(vars, opts)


    @classmethod
    def from_dict(cls, fltr_dict: Dict) -> 'ToCFilter':
        """
        Create a ToCFilter from a dictionary.

        Args:
            fltr_dict (Dict): The dictionary containing filter configuration.
            data_obj (Optional[Union[ParagraphInfo, LineInfo, List[LineInfo]]]): The data object for initializing filters.

        Returns:
            ToCFilter: The created ToCFilter.
        """
        lvl = fltr_dict.get('level')
        if lvl is None:
            raise ValueError("filter's 'level' is not set")
        if lvl < 1:
            raise ValueError("filter's 'level' must be >= 1")

        opts = ToCFilterOptions(
            check_font=fltr_dict.get('check_font', True),
            check_bbox=fltr_dict.get('check_bbox', True),
            greedy=fltr_dict.get('greedy', False)
        )

        font_filter = FontFilter.from_dict(fltr_dict.get('font', {}))
        bbox_filter = BoundingBoxFilter.from_dict(fltr_dict.get('bbox', {}))

        vars = ToCFilterVars(
            level=lvl,
            font=font_filter,
            bbox=bbox_filter
        )

        return cls(vars, opts)

    def admits(self, line: LineInfo) -> bool:
        """
        Check if the filter admits the given LineInfo object.

        Args:
            line (LineInfo): The LineInfo object.

        Returns:
            bool: True if the LineInfo object is admitted, False otherwise.
        """
        if self.opts.check_font and not self.vars.font.admits(line):
            return False
        if self.opts.check_bbox and not self.vars.bbox.admits(line.bbox):
            return False
        return True

    def __repr__(self):
        return (f"ToCFilter(vars={self.vars}, opts={self.opts})")