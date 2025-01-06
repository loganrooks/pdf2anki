"""Filter on span dictionaries

This module contains the internal representation of heading filters, which are
used to test if a span should be included in the ToC.
"""
import re
from typing import Dict, Optional, Set, Tuple, Union, List, overload, override
from pdf2anki.elements import ParagraphInfo, LineInfo
from dataclasses import asdict, dataclass, field
from multipledispatch import dispatch

from pdf2anki.utils import contained_in_bbox, get_average, is_valid_arg

DEF_TOLERANCE: dict = {"font": 1e-1, "bbox": 1e-1, "text": 1e-1}

@dataclass
class ToCEntry:
    level: int
    title: str
    pagenum: int
    bbox: Tuple[float, float, float, float]
    page_range: Optional[list[int]] = None
    start_vpos: Optional[float] = None
    end_vpos: Optional[float] = None
    text_filter_ids: Optional[Set[int]] = None
    text: Optional[list[str]] = None
    subsections: Optional[list["ToCEntry"]] = None

    def __repr__(self):
        return f"Level: {self.level}, Title: {self.title}, Pages: {self.page_range}\nText Sample: {self.text[:30] if self.text is not None else None}\nSubsections: {[f'{subsection.title}' for subsection in self.subsections if self.subsections is not None]})"

@dataclass(frozen=True)
class FontFilterOptions:
    """Options for configuring the FontFilter."""
    check_names: bool = True
    check_colors: bool = True
    check_size: bool = True
    check_width: bool = True
    check_is_upper: bool = False
    names_set_strict_equality: bool = True
    colors_set_strict_equality: bool = True
    size_tolerance: float = DEF_TOLERANCE["font"]
    ign_mask: int = 0
    ign_pattern: Optional[re.Pattern] = None

@dataclass(frozen=True)
class FontFilterVars:
    """Variables for the FontFilter."""
    names: Optional[frozenset[str]] = None
    colors: Optional[frozenset[str]] = None
    font_size: Optional[float] = None
    char_width: Optional[float] = None
    is_upper: Optional[bool] = None
   

@dataclass(frozen=True)
class BoundingBoxFilterOptions:
    """Options for configuring the BoundingBoxFilter."""
    check_left: bool = True
    check_top: bool = True
    check_right: bool = True
    check_bottom: bool = True
    require_equality: bool = False
    tolerance: float = DEF_TOLERANCE["bbox"]

@dataclass(frozen=True)
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
    
    @overload
    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[Dict[str, Union[bool, float, Optional[re.Pattern]]]] = None) -> 'FontFilter':
        ...

    @overload
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[FontFilterOptions] = None) -> 'FontFilter':
        ...

    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[Union[FontFilterOptions, Dict[str, Union[bool, float, Optional[re.Pattern]]]]] = None) -> 'FontFilter':
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
        elif is_valid_arg(opts, Dict[str, Union[bool, float, Optional[re.Pattern]]]):
            opts = FontFilterOptions(**opts)
        vars = FontFilterVars( 
            names = paragraph_info.fonts,
            colors = paragraph_info.colors,
            font_size = paragraph_info.font_size,
            char_width = paragraph_info.char_width,
            is_upper = paragraph_info.text.isupper() if opts.ign_pattern is None else re.sub(opts.ign_pattern, "", paragraph_info.text).isupper()
        )
        return cls(vars, opts)
    
    @overload
    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[FontFilterOptions] = None) -> 'FontFilter':
        ...

    @overload
    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[Dict[str, Union[bool, float, Optional[re.Pattern]]]] = None) -> 'FontFilter':
        ...

    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[Union[FontFilterOptions, Dict[str, Union[bool, float, Optional[re.Pattern]]]]] = None) -> 'FontFilter':
        """
        Create a FontFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            opts (Optional[Union[FontFilterOptions, Dict[str, Union[bool, float, Optional[re.Pattern]]]]]): The options for configuring the filter.

        Returns:
            FontFilter: The created FontFilter.
        """

        if opts is None:
            opts = FontFilterOptions()
        elif is_valid_arg(opts, Dict[str, Union[bool, float, Optional[re.Pattern]]]):
            opts = FontFilterOptions(**opts)
        vars = FontFilterVars(
            names=line_info.fonts,
            colors=line_info.colors,
            font_size=line_info.font_size,
            char_width=line_info.char_width,
            is_upper=line_info.text.isupper() if opts.ign_pattern is None else re.sub(opts.ign_pattern, "", line_info.text).isupper()
        )
        return cls(vars, opts)
    

    @overload
    @classmethod
    def from_line_info_list(cls, line_info_list: List[LineInfo], opts: Optional[Dict[str, Union[bool, float, Optional[re.Pattern]]]] = None) -> 'FontFilter':
        ...

    @classmethod
    def from_line_info_list(cls, line_info_list: List[LineInfo], opts: Optional[Union[FontFilterOptions, Dict[str, Union[bool, float, Optional[re.Pattern]]]]] = None) -> 'FontFilter':
        if opts is None:
            opts = FontFilterOptions()
        elif isinstance(opts, dict):
            opts = FontFilterOptions(**opts)
        vars = FontFilterVars(
            names=set().union(*(line_info.fonts for line_info in line_info_list)),
            colors=set().union(*(line_info.colors for line_info in line_info_list)),
            font_size=get_average([line_info.font_size for line_info in line_info_list]),
            char_width=get_average([line_info.char_width for line_info in line_info_list]),
            is_upper=all([line_info.text.isupper() for line_info in line_info_list]) if opts.ign_pattern is None 
                    else all([re.sub(opts.ign_pattern, "", line_info.text).isupper() for line_info in line_info_list])
        )
        return cls(vars, opts)
    
    @overload
    @classmethod
    def from_dict(cls, fltr_dict: Dict[str, Union[str, float, bool]]) -> 'FontFilter':
        ...

    @overload
    @classmethod
    def from_dict(cls, fltr_dict: Dict[str, Union[str, float, bool]], opts: Optional[FontFilterOptions]) -> 'FontFilter':
        ...

    @classmethod
    def from_dict(cls, fltr_dict: Dict[str, Union[str, float, bool]], opts: Optional[FontFilterOptions] = None) -> 'FontFilter':
        if opts is None:
            opts = FontFilterOptions(
                check_names=fltr_dict.get('check_name', True),
                check_colors=fltr_dict.get('check_color', True),
                check_size=fltr_dict.get('check_size', True),
                check_width=fltr_dict.get('check_width', True),
                check_is_upper=fltr_dict.get('check_is_upper', False),
                names_set_strict_equality=fltr_dict.get('names_set_strict_equality', True),
                colors_set_strict_equality=fltr_dict.get('colors_set_strict_equality', True),
                size_tolerance=fltr_dict.get('size_tolerance', DEF_TOLERANCE["font"]),
                ign_pattern=fltr_dict.get('ign_pattern')
            )
        vars = FontFilterVars(
            names=fltr_dict.get('names'),
            colors=fltr_dict.get('colors'),
            font_size=fltr_dict.get('font_size'),
            char_width=fltr_dict.get('char_width'),
            is_upper=fltr_dict.get('is_upper')
        )
        return cls(vars, opts)
    
    @dispatch(LineInfo)
    def admits(self, line_info: LineInfo) -> bool:
        """
        Check if a LineInfo object is admitted by the filter.

        Args:
            line_info (LineInfo): The LineInfo object.

        Returns:
            bool: True if the LineInfo object is admitted, False otherwise.
        """
        if self.opts.check_names and not (self.vars.names == line_info.fonts \
                                            if self.opts.names_set_strict_equality \
                                                else self.vars.names.issubset(line_info.fonts)):
            return False
        if self.opts.check_colors and not (self.vars.colors == line_info.colors \
                                            if self.opts.colors_set_strict_equality \
                                                else self.vars.colors.issubset(line_info.colors)):
            return False
        if self.opts.check_size and not admits_float(self.vars.font_size, line_info.font_size, self.opts.size_tolerance):
            return False
        if self.opts.check_width and not admits_float(self.vars.char_width, line_info.char_width, self.opts.size_tolerance):
            return False
        if self.opts.check_is_upper and not (self.vars.is_upper == line_info.text.isupper() if self.opts.ign_pattern is None \
                else re.sub(self.opts.ign_pattern, "", line_info.text).isupper()):
            return False
        return True
    
    @dispatch(ParagraphInfo)
    def admits(self, paragraph_info: ParagraphInfo) -> bool:
        """
        Check if a ParagraphInfo object is admitted by the filter.

        Args:
            paragraph_info (ParagraphInfo): The ParagraphInfo object.

        Returns:
            bool: True if the ParagraphInfo object is admitted, False otherwise.
        """
        if self.opts.check_names and not (self.vars.names == paragraph_info.fonts \
                                            if self.opts.names_set_strict_equality \
                                                else self.vars.names.issubset(paragraph_info.fonts)):
            return False
        if self.opts.check_colors and not (self.vars.colors == paragraph_info.colors \
                                            if self.opts.colors_set_strict_equality \
                                                else self.vars.colors.issubset(paragraph_info.colors)):
            return False
        if self.opts.check_size and not admits_float(self.vars.font_size, paragraph_info.font_size, self.opts.size_tolerance):
            return False
        if self.opts.check_width and not admits_float(self.vars.char_width, paragraph_info.char_width, self.opts.size_tolerance):
            return False
        if self.opts.check_is_upper and not (self.vars.is_upper == paragraph_info.text.isupper() if self.opts.ign_pattern is None \
                else re.sub(self.opts.ign_pattern, "", paragraph_info.text).isupper()):
            return False
        return True
    
    def __hash__(self):
        return hash((self.vars, self.opts))
    
    def __repr__(self):
        return (f"FontFilter(vars={self.vars},\nopts={self.opts})\n")


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
    
    def __repr__(self):
        return (f"BoundingBoxFilter(\n\tvars={self.vars},\n\topts={self.opts})\n".replace(", ", ",\n\t\t"))

    @overload
    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[BoundingBoxFilterOptions] = None) -> 'BoundingBoxFilter':
        ...

    @overload
    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[Dict[str, Union[bool, float]]] = None) -> 'BoundingBoxFilter':
        ...

    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[Union[BoundingBoxFilterOptions, Dict[str, Union[bool, float]]]] = None) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a ParagraphInfo object.

        Args:
            paragraph_info (ParagraphInfo): The ParagraphInfo object.
            opts (Optional[Union[BoundingBoxFilterOptions, Dict[str, Union[bool, float]]]]): The options for configuring the filter.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """
        if opts is None:
            opts = BoundingBoxFilterOptions()
        elif is_valid_arg(opts, Dict[str, Union[bool, float]]):
            opts = BoundingBoxFilterOptions(**opts)
        vars = BoundingBoxFilterVars(
            left=paragraph_info.bbox[0],
            bottom=paragraph_info.bbox[1],
            right=paragraph_info.bbox[2],
            top=paragraph_info.bbox[3]
        )
        return cls(vars, opts)

    @overload
    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[BoundingBoxFilterOptions] = None) -> 'BoundingBoxFilter':
        ...

    @overload
    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[Dict[str, Union[bool, float]]] = None) -> 'BoundingBoxFilter':
        ...

    @classmethod
    def from_line_info(cls, line_info: LineInfo, opts: Optional[Union[BoundingBoxFilterOptions, Dict[str, Union[bool, float]]]] = None) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            opts (Optional[Union[BoundingBoxFilterOptions, Dict[str, Union[bool, float]]]]): The options for configuring the filter.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """
        if opts is None:
            opts = BoundingBoxFilterOptions()
        elif is_valid_arg(opts, Dict[str, Union[bool, float]]):
            opts = BoundingBoxFilterOptions(**opts)
        vars = BoundingBoxFilterVars(
            left=line_info.bbox[0],
            bottom=line_info.bbox[1],
            right=line_info.bbox[2],
            top=line_info.bbox[3]
        )
        return cls(vars, opts)

    @overload
    @classmethod
    def from_line_info_list(cls, line_info_list: List[LineInfo], opts: Optional[BoundingBoxFilterOptions] = None) -> 'BoundingBoxFilter':
        ...

    @overload
    @classmethod
    def from_line_info_list(cls, line_info_list: List[LineInfo], opts: Optional[Dict[str, Union[bool, float]]] = None) -> 'BoundingBoxFilter':
        ...

    @classmethod
    def from_line_info_list(cls, line_info_list: List[LineInfo], opts: Optional[Union[BoundingBoxFilterOptions, Dict[str, Union[bool, float]]]] = None) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a list of LineInfo objects.

        Args:
            line_info_list (List[LineInfo]): The list of LineInfo objects.
            opts (Optional[Union[BoundingBoxFilterOptions, Dict[str, Union[bool, float]]]]): The options for configuring the filter.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """
        if opts is None:
            opts = BoundingBoxFilterOptions()
        elif isinstance(opts, dict):
            opts = BoundingBoxFilterOptions(**opts)
        vars = BoundingBoxFilterVars(
            left=min(line_info.bbox[0] for line_info in line_info_list),
            bottom=min(line_info.bbox[1] for line_info in line_info_list),
            right=max(line_info.bbox[2] for line_info in line_info_list),
            top=max(line_info.bbox[3] for line_info in line_info_list)
        )
        return cls(vars, opts)
    
    @overload
    @classmethod
    def from_dict(cls, fltr_dict: Dict[str, float]) -> 'BoundingBoxFilter':
        ...
    
    @overload
    @classmethod
    def from_dict(cls, fltr_dict: Dict[str, float], opts: Optional[BoundingBoxFilterOptions]) -> 'BoundingBoxFilter':
        ...
    
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
    
    @overload
    @classmethod
    def from_tuple(cls, bbox: Tuple[float, float, float, float], opts: Optional[BoundingBoxFilterOptions] = None) -> 'BoundingBoxFilter': 
        ...

    @overload
    @classmethod
    def from_tuple(cls, bbox: Tuple[float, float, float, float], opts: Optional[Dict[str, Union[bool, float]]] = None) -> 'BoundingBoxFilter':
        ...

    @classmethod
    def from_tuple(cls, bbox: Tuple[float, float, float, float], opts: Optional[Union[BoundingBoxFilterOptions, Dict[str, Union[bool, float]]]] = None) -> 'BoundingBoxFilter':
        """
        Create a BoundingBoxFilter from a tuple.

        Args:
            bbox (Tuple[float, float, float, float]): The bounding box.
            opts (Optional[Union[BoundingBoxFilterOptions, Dict[str, Union[bool, float]]]]): The options for configuring the filter.

        Returns:
            BoundingBoxFilter: The created BoundingBoxFilter.
        """
        if opts is None:
            opts = BoundingBoxFilterOptions()
        elif isinstance(opts, dict):
            opts = BoundingBoxFilterOptions(**opts)
        vars = BoundingBoxFilterVars(
            left=bbox[0],
            bottom=bbox[1],
            right=bbox[2],
            top=bbox[3]
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
        if self.opts.require_equality:
            if self.opts.check_left and not admits_float(self.vars.left, bbox[0], self.opts.tolerance):
                return False
            if self.opts.check_bottom and not admits_float(self.vars.bottom, bbox[1], self.opts.tolerance):
                return False
            if self.opts.check_right and not admits_float(self.vars.right, bbox[2], self.opts.tolerance):
                return False
            if self.opts.check_top and not admits_float(self.vars.top, bbox[3], self.opts.tolerance):
                return False
        else:
            if not contained_in_bbox(bbox, (self.vars.left, self.vars.bottom, self.vars.right, self.vars.top), 1 - self.opts.tolerance):
                return False
        return True

    def __hash__(self):
        return hash((self.vars, self.opts))
    

@dataclass(frozen=True)
class TextFilterOptions:
    """Options for configuring the TextFilter."""
    check_font: bool = True
    check_bbox: bool = False
    check_header: bool = False
    tolerance: float = DEF_TOLERANCE["text"]

@dataclass(frozen=True)
class TextFilterVars:
    """Variables for the TextFilter."""
    font: FontFilter
    bbox: Optional[BoundingBoxFilter] = None
    header: Optional[ToCEntry] = None

class TextFilter:
    def __init__(self, vars: TextFilterVars, opts: TextFilterOptions):
        """
        Initialize a TextFilter.

        Args:
            vars (TextFilterVars): The variables for the filter.
            opts (TextFilterOptions): The options for configuring the filter.
        """
        self.vars = vars
        self.opts = opts

    def __repr__(self):
        return (f"TextFilter(\n\tvars={self.vars},\n\topts={self.opts})\n".replace(", ", ",\n\t\t"))

    def __hash__(self):
        # Conditionally include bbox and font in the hash based on options
        vars_to_hash = (self.vars.header,)
        if self.opts.check_font:
            vars_to_hash += (self.vars.font,)
        if self.opts.check_bbox:
            vars_to_hash += (self.vars.bbox,)
        return hash((vars_to_hash, self.opts))

    def __eq__(self, other):
        if not isinstance(other, TextFilter):
            return False
        # Conditionally include bbox and font in the equality check based on options
        if self.opts.check_font and self.vars.font != other.vars.font:
            return False
        if self.opts.check_bbox and self.vars.bbox != other.vars.bbox:
            return False
        return self.vars.header == other.vars.header and self.opts == other.opts

    @overload
    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[TextFilterOptions] = None, font_opts: Optional[FontFilterOptions] = None, bbox_opts: Optional[BoundingBoxFilterOptions] = None, header: Optional[ToCEntry] = None) -> 'TextFilter':
        ...

    @overload
    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[Dict[str, Union[TextFilterOptions, FontFilterOptions, BoundingBoxFilterOptions]]] = None, header: Optional[ToCEntry] = None) -> 'TextFilter':
        ...
    
    @overload
    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[Dict[str, Union[bool, float, FontFilterOptions, BoundingBoxFilterOptions]]] = None, header: Optional[ToCEntry] = None) -> 'TextFilter':
        ...

    @override
    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, opts: Optional[Union[TextFilterOptions, 
                                                                                     Dict[str, Union[FontFilterOptions, TextFilterOptions, BoundingBoxFilterOptions]], 
                                                                                     Dict[str, Union[bool, float, FontFilterOptions, BoundingBoxFilterOptions]]]] = None, 
                                                                                     font_opts: Optional[FontFilterOptions] = None,
                                                                                     bbox_opts: Optional[BoundingBoxFilterOptions] = None,
                                                                                     header: Optional[ToCEntry] = None) -> 'TextFilter':
        """
        Create a TextFilter from a ParagraphInfo object.

        Args:
            paragraph_info (ParagraphInfo): The ParagraphInfo object.
            fltr_dict (Dict): The dictionary containing filter configuration.

        Returns:
            TextFilter: The created TextFilter.
        """
        if opts is None:
            opts = TextFilterOptions(check_bbox=False)

        elif is_valid_arg(opts, Dict[str, Optional[Union[bool, float, re.Pattern]]]):
            font_opts = opts.get('font', None)
            bbox_opts = opts.get('bbox', None)
            opts = TextFilterOptions(**opts.get('text', {"check_bbox": False}))

        elif is_valid_arg(opts, Dict[str, Optional[Union[bool, float, FontFilterOptions, BoundingBoxFilterOptions]]]):
            font_opts = opts.get('font', None)
            bbox_opts = opts.get('bbox', None)
            opts = TextFilterOptions(
                check_font=opts.get('check_font', True),
                check_bbox=opts.get('check_bbox', False),
                check_header=opts.get('check_header', False),
                tolerance=opts.get('tolerance', DEF_TOLERANCE["text"])
            )
        elif is_valid_arg(opts, Dict[str, Optional[Union[FontFilterOptions, TextFilterOptions, BoundingBoxFilterOptions]]]):
            font_opts = opts.get('font', None)
            bbox_opts = opts.get('bbox', None)
            opts = opts.get('text', None)

        assert isinstance(font_opts, FontFilterOptions) or font_opts is None
        assert isinstance(bbox_opts, BoundingBoxFilterOptions) or bbox_opts is None
        assert isinstance(header, ToCEntry) or header is None

        # if the fail all the arg checks, then we know that opts is a TextFilterOptions object
        font_filter = FontFilter.from_paragraph_info(paragraph_info, font_opts)
        bbox_filter = BoundingBoxFilter.from_paragraph_info(paragraph_info, bbox_opts)

        vars = TextFilterVars(
            font=font_filter,
            bbox=bbox_filter,
            header=header
        )

        return cls(vars, opts)
    
    @classmethod
    def from_dict(cls, fltr_dict: Dict[str, Union[bool, FontFilterOptions, BoundingBoxFilterOptions, ToCEntry]]) -> 'TextFilter':
        """
        Create a TextFilter from a dictionary.

        Args:
            fltr_dict (Dict): The dictionary containing filter configuration.

        Returns:
            TextFilter: The created TextFilter.
        """
        opts = TextFilterOptions(
            check_font=fltr_dict.get('check_font', True),
            check_bbox=fltr_dict.get('check_bbox', False),
            check_header=fltr_dict.get('check_header', False),
            tolerance=fltr_dict.get('tolerance', DEF_TOLERANCE["text"])
        )

        font_opts = fltr_dict.get('font_opts', {})
        bbox_opts = fltr_dict.get('bbox_opts', {})
        header = fltr_dict.get('header', {})

        font_filter = FontFilter.from_dict(fltr_dict.get('font', {}), font_opts)
        bbox_filter = BoundingBoxFilter.from_dict(fltr_dict.get('bbox', {}), bbox_opts)

        vars = TextFilterVars(
            font=font_filter,
            bbox=bbox_filter,
            header=header
        )

        return cls(vars, opts)
    
    def _admits_header(self, paragraph: ParagraphInfo, tolerance: float = DEF_TOLERANCE['text']) -> bool:
        """
        Check if the filter admits the given ParagraphInfo object as belonging to a specific header.

        Args:
            paragraph (ParagraphInfo): The ParagraphInfo object.

        Returns:
            bool: True if the ParagraphInfo object is admitted, False otherwise.
        """

        if self.vars.header is None:
            return False
        if not paragraph.pagenum in range(self.vars.header.page_range[0], self.vars.header.page_range[1] + 1):
            return False
        if paragraph.pagenum == self.vars.header.page_range[0] and paragraph.bbox[3] - self.vars.header.start_vpos > tolerance:
            return False
        if paragraph.pagenum == self.vars.header.page_range[1] and self.vars.header.end_vpos - paragraph.bbox[1] > tolerance:
            return False
        return True
    
    def admits(self, paragraph: ParagraphInfo) -> bool:
        """
        Check if the filter admits the given LineInfo object.

        Args:
            line (LineInfo): The LineInfo object.

        Returns:
            bool: True if the LineInfo object is admitted, False otherwise.
        """
        if self.opts.check_font and not self.vars.font.admits(paragraph):
            return False
        if self.opts.check_bbox and not self.vars.bbox.admits(paragraph.bbox):
            return False
        if self.opts.check_header and not self._admits_header(paragraph, self.opts.tolerance):
            return False
        return True
    
    def __repr__(self):
        return (f"TextFilter((\n\tvars={self.vars},\n\topts={self.opts})\n".replace(", ", ",\n\t\t"))
    

@dataclass(frozen=True)
class ToCFilterOptions:
    """Options for configuring the ToCFilter."""
    check_font: bool = True
    check_bbox: bool = True
    greedy: bool = False

@dataclass(frozen=True)
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

    def __hash__(self):
        # Conditionally include bbox and font in the hash based on options
        vars_to_hash = (self.vars.level,)
        if self.opts.check_font:
            vars_to_hash += (self.vars.font,)
        if self.opts.check_bbox:
            vars_to_hash += (self.vars.bbox,)
        return hash((vars_to_hash, self.opts))

    def __eq__(self, other):
        if not isinstance(other, TextFilter):
            return False
        # Conditionally include bbox and font in the equality check based on options
        if self.opts.check_font and self.vars.font != other.vars.font:
            return False
        if self.opts.check_bbox and self.vars.bbox != other.vars.bbox:
            return False
        return self.vars.level == other.vars.level and self.opts == other.opts

    @classmethod
    def from_paragraph_info(cls, paragraph_info: ParagraphInfo, 
                            level: int,
                            opts: dict[str, Union[bool, FontFilterOptions, BoundingBoxFilterOptions]]) -> 'ToCFilter':
        """
        Create a ToCFilter from a ParagraphInfo object.

        Args:
            paragraph_info (ParagraphInfo): The ParagraphInfo object.
            fltr_dict (Dict): The dictionary containing filter configuration.

        Returns:
            ToCFilter: The created ToCFilter.
        """
        if level < 1:
            raise ValueError("filter's 'level' must be >= 1")
        
        if opts is None:
            opts = {"toc": ToCFilterOptions(), "font": FontFilterOptions(), "bbox": BoundingBoxFilterOptions()}
        elif is_valid_arg(opts, Dict[str, Union[bool, FontFilterOptions, BoundingBoxFilterOptions]]):
            font_opts = opts.get('font', FontFilterOptions())
            bbox_opts = opts.get('bbox', BoundingBoxFilterOptions())
            opts = ToCFilterOptions(
                    check_font=opts.get('check_font', True),
                    check_bbox=opts.get('check_bbox', True),
                    greedy=opts.get('greedy', False)
            )
        elif is_valid_arg(opts, Dict[str, Dict[str, Union[str, float, bool]]]):
            font_opts = FontFilterOptions(**opts.get('font', {}))
            bbox_opts = BoundingBoxFilterOptions(**opts.get('bbox', {}))
            opts = ToCFilterOptions(**opts.get('toc', {}))
        
        elif is_valid_arg(opts, Dict[str, Optional[Union[ToCFilterOptions, FontFilterOptions, BoundingBoxFilterOptions]]]):
            font_opts = opts.get('font', FontFilterOptions())
            bbox_opts = opts.get('bbox', BoundingBoxFilterOptions())
            opts = opts.get('toc', ToCFilterOptions())
        else:
            raise ValueError("Invalid argument types for 'opts'")
        
        font_filter = FontFilter.from_paragraph_info(paragraph_info, font_opts)
        bbox_filter = BoundingBoxFilter.from_paragraph_info(paragraph_info, bbox_opts)

        vars = ToCFilterVars(
            level=level,
            font=font_filter,
            bbox=bbox_filter
        )

        return cls(vars, opts)
    
    @overload
    @classmethod
    def from_line_info(cls, line_info: LineInfo, 
                       level: int, 
                       opts: Optional[Dict[str, Union[bool, FontFilterOptions, BoundingBoxFilterOptions]]] = None) -> 'ToCFilter': 
        ...

    @overload
    @classmethod
    def from_line_info(cls, line_info: LineInfo,
                       level: int, 
                       opts: Optional[Dict[str, Dict[str, Union[str, int, bool]]]] = None) -> 'ToCFilter': 
        ...

    @overload
    @classmethod
    def from_line_info(cls, line_info: LineInfo,
                       level: int, 
                       opts: Optional[Dict[str, Union[ToCFilterOptions, FontFilterOptions, BoundingBoxFilterOptions]]] = None) -> 'ToCFilter': 
        ...

    @classmethod
    def from_line_info(cls, line_info: LineInfo,
                       level: int, 
                       opts: Optional[Dict[str, Union[Dict[str, Union[str, float, bool]], bool, FontFilterOptions, BoundingBoxFilterOptions, ToCFilterOptions]]] = None) -> 'ToCFilter':
        """
        Create a ToCFilter from a LineInfo object.

        Args:
            line_info (LineInfo): The LineInfo object.
            level (int): The level of the ToCFilter.
            opts (Optional[Union[Dict[str, Union[str, float, bool]], FontFilterOptions, BoundingBoxFilterOptions, ToCFilterOptions]]): The options for configuring the filter.

        Returns:
            ToCFilter: The created ToCFilter.
        """
        if level < 1:
            raise ValueError("filter's 'level' must be >= 1")
        
        if opts is None:
            opts = {"toc": ToCFilterOptions(), "font": FontFilterOptions(), "bbox": BoundingBoxFilterOptions()}
        elif is_valid_arg(opts, Dict[str, Union[bool, FontFilterOptions, BoundingBoxFilterOptions]]):
            font_opts = opts.get('font', FontFilterOptions())
            bbox_opts = opts.get('bbox', BoundingBoxFilterOptions())
            opts = ToCFilterOptions(
                    check_font=opts.get('check_font', True),
                    check_bbox=opts.get('check_bbox', True),
                    greedy=opts.get('greedy', False)
            )
        elif is_valid_arg(opts, Dict[str, Dict[str, Union[str, float, bool]]]):
            font_opts = FontFilterOptions(**opts.get('font', {}))
            bbox_opts = BoundingBoxFilterOptions(**opts.get('bbox', {}))
            opts = ToCFilterOptions(**opts.get('toc', {}))
        
        elif is_valid_arg(opts, Dict[str, Optional[Union[ToCFilterOptions, FontFilterOptions, BoundingBoxFilterOptions]]]):
            font_opts = opts.get('font', FontFilterOptions())
            bbox_opts = opts.get('bbox', BoundingBoxFilterOptions())
            opts = opts.get('toc', ToCFilterOptions())
        else:
            raise ValueError("Invalid argument types for 'opts'")

        font_filter = FontFilter.from_line_info(line_info, font_opts)
        bbox_filter = BoundingBoxFilter.from_line_info(line_info, bbox_opts)

        vars = ToCFilterVars(
            level=level,
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

    def __hash__(self):
        return hash((self.vars, self.opts))
    
    def __repr__(self):
        return (f"ToCFilter(\n\tvars={self.vars},\n\topts={self.opts})\n".replace(", ", ",\n\t\t"))



def is_similar_text_filter(tf1: TextFilter, tf2: TextFilter, tolerance_dict: Dict[str, float]) -> bool:
    """
    Check if two TextFilters are similar based on their options and a tolerance dictionary.

    Args:
        tf1 (TextFilter): The first TextFilter.
        tf2 (TextFilter): The second TextFilter.
        tolerance_dict (Dict[str, float]): The tolerance dictionary with keys "bbox" and "font".

    Returns:
        bool: True if the TextFilters are similar, False otherwise.
    """
    # Check if options are equal
    if tf1.opts != tf2.opts:
        return False

    # Check font similarity if check_font is True
    if tf1.opts.check_font:
        font_tolerance = tolerance_dict.get("font", 0.0)
        if not admits_float(tf1.vars.font.vars.font_size, tf2.vars.font.vars.font_size, font_tolerance * tf1.vars.font.vars.font_size):
            return False
        if not admits_float(tf1.vars.font.vars.char_width, tf2.vars.font.vars.char_width, font_tolerance * tf1.vars.font.vars.char_width):
            return False

    # Check bbox similarity if check_bbox is True
    if tf1.opts.check_bbox:
        bbox_tolerance = tolerance_dict.get("bbox", 0.0)
        if not admits_float(tf1.vars.bbox.vars.left, tf2.vars.bbox.vars.left, bbox_tolerance):
            return False
        if not admits_float(tf1.vars.bbox.vars.bottom, tf2.vars.bbox.vars.bottom, bbox_tolerance):
            return False
        if not admits_float(tf1.vars.bbox.vars.right, tf2.vars.bbox.vars.right, bbox_tolerance):
            return False
        if not admits_float(tf1.vars.bbox.vars.top, tf2.vars.bbox.vars.top, bbox_tolerance):
            return False
    return True

def find_similar_filter_sets(filters: List[TextFilter], tolerance_dict: Dict[str, float]) -> List[Set[int]]:
    """
    Find sets of similar TextFilters based on their options and a tolerance dictionary.

    Args:
        filters (List[TextFilter]): The list of TextFilters.
        tolerance_dict (Dict[str, float]): The tolerance dictionary with keys "bbox" and "font".

    Returns:
        List[Set[int]]: A list of sets, where each set contains indices of similar TextFilters.
    """
    similar_sets: list[set] = []
    for i, tf1 in enumerate(filters):
        found = False
        for similar_set in similar_sets:
            if any(is_similar_text_filter(tf1, filters[j], tolerance_dict) for j in similar_set):
                similar_set.add(i)
                found = True
                break
        if not found:
            similar_sets.append({i})
    return similar_sets

def average_float_vars(filters: List[TextFilter], indices: Set[int]) -> TextFilter:
    """
    Average the float type variables of a set of similar TextFilters.

    Args:
        filters (List[TextFilter]): The list of TextFilters.
        indices (Set[int]): The indices of the similar TextFilters to average.

    Returns:
        TextFilter: The combined TextFilter with averaged float type variables.
    """
    if not indices:
        raise ValueError("Indices set cannot be empty")

    # Initialize sums and counts
    font_size_sum = 0.0
    char_width_sum = 0.0
    bbox_left_sum = 0.0
    bbox_top_sum = 0.0
    bbox_right_sum = 0.0
    bbox_bottom_sum = 0.0
    count = len(indices)

    # Sum the float type variables
    for i in indices:
        tf = filters[i]
        if tf.opts.check_font:
            font_size_sum += tf.vars.font.vars.font_size or 0.0
            char_width_sum += tf.vars.font.vars.char_width or 0.0
        if tf.opts.check_bbox:
            bbox_left_sum += tf.vars.bbox.vars.left or 0.0
            bbox_top_sum += tf.vars.bbox.vars.top or 0.0
            bbox_right_sum += tf.vars.bbox.vars.right or 0.0
            bbox_bottom_sum += tf.vars.bbox.vars.bottom or 0.0

    # Calculate averages
    averaged_font_size = font_size_sum / count if count > 0 else None
    averaged_char_width = char_width_sum / count if count > 0 else None
    averaged_bbox_left = bbox_left_sum / count if count > 0 else None
    averaged_bbox_top = bbox_top_sum / count if count > 0 else None
    averaged_bbox_right = bbox_right_sum / count if count > 0 else None
    averaged_bbox_bottom = bbox_bottom_sum / count if count > 0 else None

    # Create a new TextFilter with averaged values
    averaged_font_filter_vars = FontFilterVars(
        font_size=averaged_font_size,
        char_width=averaged_char_width,
        names=filters[next(iter(indices))].vars.font.vars.names,
        colors=filters[next(iter(indices))].vars.font.vars.colors,
        is_upper=filters[next(iter(indices))].vars.font.vars.is_upper
    )

    averaged_bbox_filter_vars = BoundingBoxFilterVars(
        left=averaged_bbox_left,
        top=averaged_bbox_top,
        right=averaged_bbox_right,
        bottom=averaged_bbox_bottom
    )

    averaged_font_filter = FontFilter(averaged_font_filter_vars, filters[next(iter(indices))].vars.font.opts)
    averaged_bbox_filter = BoundingBoxFilter(averaged_bbox_filter_vars, filters[next(iter(indices))].vars.bbox.opts)

    averaged_text_filter_vars = TextFilterVars(
        font=averaged_font_filter,
        bbox=averaged_bbox_filter,
        header=filters[next(iter(indices))].vars.header
    )

    return TextFilter(averaged_text_filter_vars, filters[next(iter(indices))].opts)

def merge_similar_text_filters(filters: List[TextFilter], tolerance_dict: Dict[str, float]) -> List[TextFilter]:
    """
    Merge similar TextFilters by averaging their float type variables.

    Args:
        filters (List[TextFilter]): The list of TextFilters.
        tolerance_dict (Dict[str, float]): The tolerance dictionary with keys "bbox" and "font".

    Returns:
        List[TextFilter]: The list of merged TextFilters.
    """
    similar_sets = find_similar_filter_sets(filters, tolerance_dict)
    merged_filters = [average_float_vars(filters, indices) for indices in similar_sets]
    return merged_filters