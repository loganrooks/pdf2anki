from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

class ElementType(Enum):
    CHAR = "char"
    LINE = "line"
    PARAGRAPH = "para"
    PAGE = "page"

@dataclass
class CharInfo:
    text: str = ""
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    size: float = 0.0
    height: float = 0.0
    width: float = 0.0
    font: str = ""
    color: str = ""

    def get_metadata(self) -> Dict[str, Union[str, Tuple[float, float, float, float], float]]:
        return {
            "bbox": self.bbox,
            "size": self.size,
            "height": self.height,
            "width": self.width,
            "font": self.font,
            "color": self.color
        }

@dataclass
class LineInfo:
    text: str = ""
    chars: List[CharInfo] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    font_size: float = 0.0
    char_height: float = 0.0
    char_width: float = 0.0
    fonts: Set[str] = field(default_factory=set)
    colors: Set[str] = field(default_factory=set)
    split_end_word: bool = False
    pagenum: Optional[int] = None

    def get_metadata(self) -> Dict[str, Union[str, Tuple[float, float, float, float], float, Set[str]]]:
        return {
            "bbox": self.bbox,
            "font_size": self.font_size,
            "char_height": self.char_height,
            "char_width": self.char_width,
            "fonts": self.fonts,
            "colors": self.colors
        }
    
    def update_pagenum(self, pagenum: int, recursive=True) -> None:
        self.pagenum = pagenum

@dataclass
class ParagraphInfo:
    text: str = ""
    lines: List[LineInfo] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    font_size: float = 0.0
    char_width: float = 0.0
    fonts: Set[str] = field(default_factory=set)
    colors: Set[str] = field(default_factory=set)
    split_end_line: bool = False
    is_indented: bool = False
    pagenum: Optional[int] = None

    def get_metadata(self) -> Dict[str, Union[str, Tuple[float, float, float, float], float, Set[str]]]:
        return {
            "bbox": self.bbox,
            "font_size": self.font_size,
            "char_width": self.char_width,
            "fonts": self.fonts,
            "colors": self.colors
        }
    
    def update_pagenum(self, pagenum: int, recursive: bool = True) -> None:
        self.pagenum = pagenum
        if recursive:
            for line in self.lines:
                line.pagenum = pagenum

@dataclass
class PageInfo:
    text: str = ""
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    fonts: Set[str] = field(default_factory=set)
    font_sizes: Set[float] = field(default_factory=set)
    char_widths: Set[float] = field(default_factory=set)
    colors: Set[str] = field(default_factory=set)
    paragraphs: List[ParagraphInfo] = field(default_factory=list)
    split_end_paragraph: bool = False
    pagenum: Optional[int] = None

    def get_metadata(self) -> Dict[str, Union[str, Tuple[float, float, float, float], Set[str]]]:
        return {
            "bbox": self.bbox,
            "fonts": self.fonts,
            "font_sizes": self.font_sizes,
            "char_widths": self.char_widths,
            "colors": self.colors
        }
    
    def update_pagenum(self, pagenum: int, recursive: bool = True) -> None:
        self.pagenum = pagenum
        if recursive:
            for paragraph in self.paragraphs:
                paragraph.update_pagenum(pagenum, recursive=recursive)



