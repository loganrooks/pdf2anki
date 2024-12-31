import pytest
from pdf2anki.filters import (
    FontFilter, FontFilterVars, FontFilterOptions,
    BoundingBoxFilter, BoundingBoxFilterVars, BoundingBoxFilterOptions,
    ToCFilter, ToCFilterVars, ToCFilterOptions
)

from pdf2anki.extraction import LineInfo, ParagraphInfo

def test_font_filter_admits_line():
    line = LineInfo(
        text="HelloWorld",
        font="SomeFont",
        color="black",
        size=12.0,
        char_width=6.0,
        bbox=(0, 0, 100, 50)
    )
    filter_vars = FontFilterVars(
        name="SomeFont",
        color="black",
        font_size=12.0,
        char_width=6.0,
        is_upper=False
    )
    filter_opts = FontFilterOptions()
    font_filter = FontFilter(filter_vars, filter_opts)
    assert font_filter.admits(line)

def test_font_filter_rejects_line_mismatch():
    line = LineInfo(
        text="HELLO!",
        font="OtherFont",
        color="blue",
        size=9.0,
        char_width=5.0,
        bbox=(0, 0, 100, 50)
    )
    filter_vars = FontFilterVars(
        name="SomeFont"
    )
    filter_opts = FontFilterOptions(check_name=True)
    font_filter = FontFilter(filter_vars, filter_opts)
    assert not font_filter.admits(line)

def test_bounding_box_filter_admits():
    line = LineInfo(text="", font="", color="", size=0, char_width=0, bbox=(10, 10, 50, 50))
    bbox_vars = BoundingBoxFilterVars(left=10, top=10, right=50, bottom=50)
    bbox_opts = BoundingBoxFilterOptions()
    bbox_filter = BoundingBoxFilter(bbox_vars, bbox_opts)
    assert bbox_filter.admits(line.bbox)

def test_bounding_box_filter_rejects():
    line = LineInfo(text="", font="", color="", size=0, char_width=0, bbox=(0, 0, 100, 100))
    bbox_vars = BoundingBoxFilterVars(left=10, top=10, right=50, bottom=50)
    bbox_opts = BoundingBoxFilterOptions()
    bbox_filter = BoundingBoxFilter(bbox_vars, bbox_opts)
    assert not bbox_filter.admits(line.bbox)

def test_toc_filter_admits():
    line = LineInfo(
        text="TestHeading",
        font="HeadingFont",
        color="black",
        size=14.0,
        char_width=7.0,
        bbox=(10, 10, 90, 20)
    )
    vars_font = FontFilterVars(name="HeadingFont", color="black", font_size=14.0)
    opts_font = FontFilterOptions()
    font_filter = FontFilter(vars_font, opts_font)
    bbox_vars = BoundingBoxFilterVars(left=10, top=10, right=90, bottom=20)
    bbox_opts = BoundingBoxFilterOptions()
    bbox_filter = BoundingBoxFilter(bbox_vars, bbox_opts)

    toc_vars = ToCFilterVars(level=1, font=font_filter, bbox=bbox_filter)
    toc_opts = ToCFilterOptions(check_font=True, check_bbox=True, greedy=False)
    toc_filter = ToCFilter(toc_vars, toc_opts)

    assert toc_filter.admits(line)

def test_toc_filter_rejects():
    line = LineInfo(
        text="WrongHeading",
        font="SomeOtherFont",
        color="blue",
        size=10.0,
        char_width=5.0,
        bbox=(10, 10, 90, 20)
    )
    vars_font = FontFilterVars(name="HeadingFont")
    opts_font = FontFilterOptions(check_name=True)
    font_filter = FontFilter(vars_font, opts_font)
    bbox_vars = BoundingBoxFilterVars(left=10, top=10, right=90, bottom=20)
    bbox_opts = BoundingBoxFilterOptions()
    bbox_filter = BoundingBoxFilter(bbox_vars, bbox_opts)

    toc_vars = ToCFilterVars(level=2, font=font_filter, bbox=bbox_filter)
    toc_opts = ToCFilterOptions(check_font=True, check_bbox=True, greedy=False)
    toc_filter = ToCFilter(toc_vars, toc_opts)

    assert not toc_filter.admits(line)
