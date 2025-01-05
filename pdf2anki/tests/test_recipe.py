from typing import List
import unittest
from unittest.mock import MagicMock
from pdf2anki.recipe import generate_recipe, Recipe
from pdf2anki.elements import PageInfo, ParagraphInfo, LineInfo, ElementType
from pdf2anki.filters import ToCFilter, TextFilter, ToCFilterOptions, TextFilterOptions, FontFilterOptions, BoundingBoxFilterOptions

import pytest
from pdf2anki.elements import PageInfo, ParagraphInfo, LineInfo
from pdf2anki.recipe import get_text_index_from_vpos

@pytest.fixture
def page_info():
    return PageInfo(
        text="First paragraph.\n\nSecond paragraph.",
        bbox=(0.0, 0.0, 100.0, 200.0),
        fonts=set(),
        font_sizes=set(),
        char_widths=set(),
        colors=set(),
        paragraphs=[
            ParagraphInfo(
                text="First paragraph.",
                lines=[
                    LineInfo(
                        text="First paragraph.",
                        chars=[],
                        bbox=(0.0, 180.0, 100.0, 200.0),
                        font_size=12.0,
                        char_height=12.0,
                        char_width=6.0,
                        fonts=set(),
                        colors=set()
                    )
                ],
                bbox=(0.0, 180.0, 100.0, 200.0),
                font_size=12.0,
                char_width=6.0,
                fonts=set(),
                colors=set()
            ),
            ParagraphInfo(
                text="Second paragraph.",
                lines=[
                    LineInfo(
                        text="Second paragraph.",
                        chars=[],
                        bbox=(0.0, 160.0, 100.0, 180.0),
                        font_size=12.0,
                        char_height=12.0,
                        char_width=6.0,
                        fonts=set(),
                        colors=set()
                    )
                ],
                bbox=(0.0, 160.0, 100.0, 180.0),
                font_size=12.0,
                char_width=6.0,
                fonts=set(),
                colors=set()
            )
        ],
        split_end_paragraph=False,
        pagenum=1
    )

def test_get_text_index_from_vpos_start_of_page(page_info):
    start_vpos = 200.0
    index = get_text_index_from_vpos(start_vpos, page_info)
    assert index == 0

def test_get_text_index_from_vpos_middle_of_page(page_info):
    start_vpos = 170.0
    index = get_text_index_from_vpos(start_vpos, page_info)
    assert index == len("First paragraph.\n\n")

def test_get_text_index_from_vpos_end_of_page(page_info):
    start_vpos = 160.0
    index = get_text_index_from_vpos(start_vpos, page_info)
    assert index == len("First paragraph.\n\nSecond paragraph.\n\n")


from unittest.mock import MagicMock
from pdf2anki.elements import PageInfo, ParagraphInfo, LineInfo

# Mock PageInfo
def create_test_doc(text_block: str, paragraphs_per_page: int, lines_per_paragraph: int) -> List[PageInfo]:
    paragraphs = text_block.split('\n\n')
    doc = []

    for page_num in range(1, (len(paragraphs) // paragraphs_per_page) + 1):
        page_paragraphs = []
        for para_num in range(paragraphs_per_page):
            if (page_num - 1) * paragraphs_per_page + para_num >= len(paragraphs):
                break
            paragraph_text = paragraphs[(page_num - 1) * paragraphs_per_page + para_num]
            lines = paragraph_text.split('\n')
            line_infos = []
            for line_num, line_text in enumerate(lines[:lines_per_paragraph]):
                line_info = LineInfo(
                    text=line_text,
                    chars=(),
                    bbox=(0.0, 0.0, 100.0, 10.0 * (line_num + 1)),
                    font_size=12.0,
                    char_height=12.0,
                    char_width=6.0,
                    fonts=frozenset({"Arial"}),
                    colors=frozenset({"#000000"})
                )
                line_infos.append(line_info)
            paragraph_info = ParagraphInfo(
                text=paragraph_text,
                lines=tuple(line_infos),
                bbox=(0.0, 0.0, 100.0, 10.0 * len(lines)),
                font_size=12.0,
                char_width=6.0,
                fonts=frozenset({"Arial"}),
                colors=frozenset({"#000000"})
            )
            page_paragraphs.append(paragraph_info)
        page_info = PageInfo(
            text='\n\n'.join(paragraph.text for paragraph in page_paragraphs),
            bbox=(0.0, 0.0, 100.0, 200.0),
            fonts=frozenset({"Arial"}),
            font_sizes=frozenset([12.0]),
            char_widths=frozenset([6.0]),
            colors=frozenset({"#000000"}),
            paragraphs=tuple(page_paragraphs),
            split_end_paragraph=False,
            pagenum=page_num
        )
        doc.append(page_info)

    return doc

# Mock PageInfo


class TestGenerateRecipe(unittest.TestCase):

    def setUp(self):
        # Use the mocked PageInfo, ParagraphInfo, LineInfo
        self.doc = create_test_doc("First paragraph.\nSecond line.\n\nSecond paragraph.\nSecond line.", 1, 2)


        # Mock headers
        self.headers = [
            {"header": (1, "First Paragraph"), "level": 1, "text": [(1, "Second line")]},
            {"header": (2, "Second Paragraph"), "level": 2, "text": [(2, "Second line")]}
        ]

        # Mock ToCFilter and TextFilter
        self.toc_filter = MagicMock(spec=ToCFilter)
        self.text_filter = MagicMock(spec=TextFilter)

        # Mock options
        self.toc_filter_options = [{"toc": ToCFilterOptions(), "font": FontFilterOptions(), "bbox": BoundingBoxFilterOptions()} for _ in self.headers]
        self.text_filter_options = [{"text": TextFilterOptions(), "font": FontFilterOptions(), "bbox": BoundingBoxFilterOptions()} for _ in self.headers]

    def test_generate_recipe_basic(self):
        recipe = generate_recipe(self.doc, self.headers)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 2)
        self.assertEqual(len(recipe.text_filters), 0)

    def test_generate_recipe_with_text_filters(self):
        recipe = generate_recipe(self.doc, self.headers, include_text_filters=True)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 2)
        self.assertEqual(len(recipe.text_filters), 1, "Expected 1 text filter, because they are duplicate filters, got %d" % len(recipe.text_filters))

    def test_generate_recipe_with_options(self):
        recipe = generate_recipe(self.doc, self.headers, 
                                 toc_filter_options=self.toc_filter_options, 
                                 text_filter_options=self.text_filter_options)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 2)
        self.assertEqual(len(recipe.text_filters), 0)

    def test_generate_recipe_with_invalid_page_numbers(self):
        with self.assertRaises(ValueError):
            generate_recipe(self.doc, self.headers, page_numbers=[-1])

    def test_generate_recipe_with_custom_tolerances(self):
        tolerances = {"font": 0.2, "bbox": 0.2}
        recipe = generate_recipe(self.doc, self.headers, tolerances=tolerances)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 2)
        self.assertEqual(len(recipe.text_filters), 0)

if __name__ == '__main__':
    unittest.main()