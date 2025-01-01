import pytest
from pdf2anki.filters import (
    FontFilter, FontFilterVars, FontFilterOptions,
    BoundingBoxFilter, BoundingBoxFilterVars, BoundingBoxFilterOptions,
    ToCFilter, ToCFilterVars, ToCFilterOptions, TextFilter, TextFilterVars, TextFilterOptions
)

from pdf2anki.extraction import LineInfo, ParagraphInfo

@pytest.fixture
def sample_line_info():
    return LineInfo(
        text="Sample Text",
        fonts=set(["Arial"]),
        colors=set(["#FFFFFF"]),
        font_size=12.0,
        char_width=6.0,
        bbox=(0, 0, 100, 50)
    )

@pytest.fixture
def text_filter():
    font_filter = FontFilter(
        vars=FontFilterVars(names={"Arial"}, colors={"#FFFFFF"}, font_size=12.0, char_width=6.0),
        opts=FontFilterOptions()
    )
    bbox_filter = BoundingBoxFilter(
        vars=BoundingBoxFilterVars(left=0, bottom=0, right=100, top=50),
        opts=BoundingBoxFilterOptions()
    )
    return TextFilter(
        vars=TextFilterVars(font=font_filter, bbox=bbox_filter),
        opts=TextFilterOptions()
    )

@pytest.mark.parametrize("fonts,colors,font_size,char_width,is_upper,expected", [
    ({"Arial"}, {"#FFFFFF"}, 12.0, 6.0, True, True),
    ({"Times New Roman"}, {"#FFFFFF"}, 12.0, 6.0, False, False),
    ({"Arial"}, {"#000000"}, 12.0, 6.0, False, False),
    ({"Arial"}, {"#FFFFFF"}, 10.0, 6.0, False, False),
    ({"Arial"}, {"#FFFFFF"}, 12.0, 1.0, False, False)
])
def test_text_filter_admits(sample_line_info, text_filter, fonts, colors, font_size, char_width, is_upper, expected):
    line = sample_line_info
    line.fonts = fonts
    line.colors = colors
    line.font_size = font_size
    line.char_width = char_width
    line.text = "SAMPLE TEST" if is_upper else "sample test"
    assert text_filter.admits(line) == expected

@pytest.mark.parametrize("case_sensitive,is_upper,expected", [
    (True, True, True),
    (True, False, False),
    (False, None, True),
    (False, True, True),
    (False, False, True)
])



def test_text_filter_case_sensitivity(sample_line_info, case_sensitive, is_upper, expected):
    font_filter = FontFilter(
        vars=FontFilterVars(names={"Arial"}, colors={"#FFFFFF"}, font_size=12.0, char_width=6.0, is_upper=is_upper),
        opts=FontFilterOptions(check_is_upper=case_sensitive)
    )
    bbox_filter = BoundingBoxFilter(
        vars=BoundingBoxFilterVars(left=0, bottom=0, right=100, top=50),
        opts=BoundingBoxFilterOptions()
    )
    text_filter = TextFilter(
        vars=TextFilterVars(font=font_filter, bbox=bbox_filter),
        opts=TextFilterOptions()
    )

    sample_line_info.text = "SAMPLE TEXT"
    assert text_filter.admits(sample_line_info) == expected

@pytest.mark.parametrize("text_input", [
    "sample boundary",
    "another sample boundary"
])
def test_text_filter_multiple_matches(sample_line_info, text_input):
    font_filter = FontFilter(
        vars=FontFilterVars(names={"Arial"}, colors={"#FFFFFF"}, font_size=12.0, char_width=6.0),
        opts=FontFilterOptions()
    )
    bbox_filter = BoundingBoxFilter(
        vars=BoundingBoxFilterVars(left=0, bottom=0, right=100, top=50),
        opts=BoundingBoxFilterOptions()
    )
    tf = TextFilter(
        vars=TextFilterVars(font=font_filter, bbox=bbox_filter),
        opts=TextFilterOptions()
    )
    sample_line_info.text = text_input
    assert tf.admits(sample_line_info) is True

@pytest.mark.parametrize("bbox_input,expected", [
    ((10, 10, 50, 50), True),
    ((-5, 10, 50, 50), False)
])
def test_bounding_box_filter_edge_cases(sample_line_info, bbox_input, expected):
    bbox_filter = BoundingBoxFilter.from_tuple(
        (9, 9, 100, 51),
        opts=BoundingBoxFilterOptions()
    )
    sample_line_info.bbox = bbox_input
    assert bbox_filter.admits(sample_line_info.bbox) == expected

def test_font_filter_ignore_color(sample_line_info):
    f_vars = FontFilterVars(names={"Arial"}, colors=None)
    f_opts = FontFilterOptions(check_colors=False)
    font_filter = FontFilter(f_vars, f_opts)
    sample_line_info.colors = "#FF00EE"
    assert font_filter.admits(sample_line_info) is True

def test_toc_filter_greedy(sample_line_info):
    tf_vars = ToCFilterVars(level=1, font=None, bbox=None)
    tf_opts = ToCFilterOptions(check_font=False, check_bbox=False, greedy=True)
    toc_filter = ToCFilter(tf_vars, tf_opts)
    sample_line_info.text = "A possible heading"
    assert toc_filter.admits(sample_line_info)

def test_font_filter_admits_line():
    line = LineInfo(
        text="HelloWorld",
        fonts="SomeFont",
        colors="black",
        font_size=12.0,
        char_width=6.0,
        bbox=(0, 0, 100, 50)
    )
    filter_vars = FontFilterVars(
        names="SomeFont",
        colors="black",
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
        fonts="OtherFont",
        colors="blue",
        font_size=9.0,
        char_width=5.0,
        bbox=(0, 0, 100, 50)
    )
    filter_vars = FontFilterVars(
        names="SomeFont"
    )
    filter_opts = FontFilterOptions(check_names=True)
    font_filter = FontFilter(filter_vars, filter_opts)
    assert not font_filter.admits(line)

def test_bounding_box_filter_admits():
    line = LineInfo(text="", fonts="", colors="", font_size=0, char_width=0, bbox=(10, 10, 50, 50))
    bbox_vars = BoundingBoxFilterVars(left=10, bottom=10, right=50, top=50)
    bbox_opts = BoundingBoxFilterOptions()
    bbox_filter = BoundingBoxFilter(bbox_vars, bbox_opts)
    assert bbox_filter.admits(line.bbox)

def test_bounding_box_filter_rejects():
    line = LineInfo(text="", fonts="", colors="", font_size=0, char_width=0, bbox=(0, 0, 100, 100))
    bbox_vars = BoundingBoxFilterVars(left=10, bottom=10, right=50, top=50)
    bbox_opts = BoundingBoxFilterOptions()
    bbox_filter = BoundingBoxFilter(bbox_vars, bbox_opts)
    assert not bbox_filter.admits(line.bbox)

def test_toc_filter_admits():
    line = LineInfo(
        text="TestHeading",
        fonts="HeadingFont",
        colors="black",
        font_size=14.0,
        char_width=7.0,
        bbox=(10, 10, 90, 20)
    )
    vars_font = FontFilterVars(names="HeadingFont", colors="black", font_size=14.0)
    opts_font = FontFilterOptions()
    font_filter = FontFilter(vars_font, opts_font)
    bbox_vars = BoundingBoxFilterVars(left=10, bottom=10, right=90, top=20)
    bbox_opts = BoundingBoxFilterOptions()
    bbox_filter = BoundingBoxFilter(bbox_vars, bbox_opts)

    toc_vars = ToCFilterVars(level=1, font=font_filter, bbox=bbox_filter)
    toc_opts = ToCFilterOptions(check_font=True, check_bbox=True, greedy=False)
    toc_filter = ToCFilter(toc_vars, toc_opts)

    assert toc_filter.admits(line)

def test_toc_filter_rejects():
    line = LineInfo(
        text="WrongHeading",
        fonts="SomeOtherFont",
        colors="blue",
        font_size=10.0,
        char_width=5.0,
        bbox=(10, 10, 90, 20)
    )
    vars_font = FontFilterVars(names="HeadingFont")
    opts_font = FontFilterOptions(check_names=True)
    font_filter = FontFilter(vars_font, opts_font)
    bbox_vars = BoundingBoxFilterVars(left=10, bottom=10, right=90, top=20)
    bbox_opts = BoundingBoxFilterOptions()
    bbox_filter = BoundingBoxFilter(bbox_vars, bbox_opts)

    toc_vars = ToCFilterVars(level=2, font=font_filter, bbox=bbox_filter)
    toc_opts = ToCFilterOptions(check_font=True, check_bbox=True, greedy=False)
    toc_filter = ToCFilter(toc_vars, toc_opts)

    assert not toc_filter.admits(line)
