from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Union, FrozenSet
import io

Primitive = Union[int, float, str, bool, None]
FileType = Union[io.BufferedIOBase, io.BufferedReader]
Element = Union["CharInfo", "LineInfo", "ParagraphInfo", "PageInfo"]

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

    def __iter__(self):
        for key, value in asdict(self).items():
            yield key, value

    def __eq__(self, other):
        if isinstance(other, CharInfo):
            return asdict(self) == asdict(other)
        return False

    def __hash__(self):
        return hash(tuple(attr for attr in self))

@dataclass
class LineInfo:
    text: str = ""
    chars: Tuple[CharInfo] = field(default_factory=tuple)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    font_size: float = 0.0
    char_height: float = 0.0
    char_width: float = 0.0
    fonts: FrozenSet[str] = field(default_factory=frozenset)
    colors: FrozenSet[str] = field(default_factory=frozenset)
    split_end_word: bool = False
    pagenum: Optional[int] = None

    def get_metadata(self) -> Dict[str, Union[str, Tuple[float, float, float, float], float, FrozenSet[str]]]:
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

    def __iter__(self):
        for key, value in asdict(self).items():
            yield key, value

    def __eq__(self, other):
        if isinstance(other, LineInfo):
            return asdict(self) == asdict(other)
        return False

    def __hash__(self):
        return hash(attr for attr in self if not attr[0] == "chars")

@dataclass
class ParagraphInfo:
    text: str = ""
    lines: Tuple[LineInfo] = field(default_factory=tuple)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    font_size: float = 0.0
    char_width: float = 0.0
    fonts: FrozenSet[str] = field(default_factory=frozenset)
    colors: FrozenSet[str] = field(default_factory=frozenset)
    split_end_line: bool = False
    is_indented: bool = False
    pagenum: Optional[int] = None

    def get_metadata(self) -> Dict[str, Union[str, Tuple[float, float, float, float], float, FrozenSet[str]]]:
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

    def __iter__(self):
        for key, value in asdict(self).items():
            yield key, value

    def __eq__(self, other):
        if isinstance(other, ParagraphInfo):
            return asdict(self) == asdict(other)
        return False

    def __hash__(self):
        return hash(attr for attr in self if not attr[0] == "lines")

@dataclass
class PageInfo:
    text: str = ""
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    fonts: FrozenSet[str] = field(default_factory=frozenset)
    font_sizes: FrozenSet[float] = field(default_factory=frozenset)
    char_widths: FrozenSet[float] = field(default_factory=frozenset)
    colors: FrozenSet[str] = field(default_factory=frozenset)
    paragraphs: Tuple[ParagraphInfo] = field(default_factory=tuple)
    split_end_paragraph: bool = False
    starts_with_indent: Optional[bool] = None
    pagenum: Optional[int] = None

    def get_metadata(self) -> Dict[str, Union[str, Tuple[float, float, float, float], FrozenSet[str]]]:
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

    def __iter__(self):
        for key, value in asdict(self).items():
            yield key, value

    def __eq__(self, other):
        if isinstance(other, PageInfo):
            return asdict(self) == asdict(other)
        return False

    def __hash__(self):
        return hash(attr for attr in self if not attr[0] == "paragraphs")

@dataclass(frozen=True)
class FileObject:
    path: str
    type: Type[FileType]
    name: str