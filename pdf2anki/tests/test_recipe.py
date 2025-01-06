from typing import List
import unittest
from unittest.mock import MagicMock
from pdf2anki.recipe import generate_recipe, Recipe
from pdf2anki.elements import PageInfo, ParagraphInfo, LineInfo, ElementType
from pdf2anki.filters import ToCFilter, TextFilter, ToCFilterOptions, TextFilterOptions, FontFilterOptions, BoundingBoxFilterOptions

import pytest
from pdf2anki.elements import PageInfo, ParagraphInfo, LineInfo
from pdf2anki.recipe import get_text_index_from_vpos
from pdf2anki.tests.utils import create_test_doc

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




class TestGenerateRecipe(unittest.TestCase):

    def setUp(self):
        # Use the mocked PageInfo, ParagraphInfo, LineInfo
        text_blocks = [
                # Page 1 has three paragraphs
                "Paragraph 1: This is the first paragraph on page 1.\nLine 2\n\n"
                "Paragraph 2: This is the second paragraph on page 1.\nLine 2\n\n"
                "Paragraph 3: This is the third paragraph on page 1.\nLine 2",
                
                # Page 2 has three paragraphs
                "Paragraph 1: This is the first paragraph on page 2.\nLine 2\n\n"
                "Paragraph 2: This is the second paragraph on page 2.\nLine 2\n\n"
                "Paragraph 3: This is the third paragraph on page 2.\nLine 2",
                
                # Page 3 has three paragraphs
                "Paragraph 1: This is the first paragraph on page 3.\nLine 2\n\n"
                "Paragraph 2: This is the second paragraph on page 3.\nLine 2\n\n"
                "Paragraph 3: This is the third paragraph on page 3.\nLine 2"
            ]
        
        headers = [
                ("Header for Page 1", 0),  # Insert at first paragraph on Page 1
                ("Header for Page 2", 1),  # Insert at second paragraph on Page 2
                ("Header for Page 3", 2)   # Insert at third paragraph on Page 3
            ]

        self.doc = create_test_doc(text_blocks, headers, text_size=12.0, header_size=16.0, font_name="Arial")


        # Mock headers for document created with create_test_doc
        self.headers = [
            {"header": (1, "Header for Page 1"), "level": 1, "text": [(1, "Paragraph 1: This is the first paragraph on page 1.")]},
            {"header": (2, "Header for Page 2"), "level": 2, "text": [(2, "Paragraph 2: This is the second paragraph on page 2.")]},
            {"header": (3, "Header for Page 3"), "level": 1, "text": [(3, "Paragraph 3: This is the third paragraph on page 3.")]}
        ]

        # Mock ToCFilter and TextFilter
        self.toc_filter = MagicMock(spec=ToCFilter)
        self.text_filter = MagicMock(spec=TextFilter)

        # Mock options
        self.toc_filter_options = [{"toc": ToCFilterOptions(), "font": FontFilterOptions(), "bbox": BoundingBoxFilterOptions()} for _ in self.headers]
        self.text_filter_options = [{"text": TextFilterOptions(check_bbox=True), "font": FontFilterOptions(), "bbox": BoundingBoxFilterOptions(require_equality=True)} for _ in self.headers]

    def test_generate_recipe_basic(self):
        recipe = generate_recipe(self.doc, self.headers)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 3)
        self.assertEqual(len(recipe.text_filters), 0)

    def test_generate_recipe_with_text_filters(self):
        recipe = generate_recipe(self.doc, self.headers, include_text_filters=True, tolerances={"font": 0.1, "bbox": 0.1}, merge_similar_text_filters=False)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 3)
        self.assertEqual(len(recipe.text_filters), 3, "Expected 3 text filter, because we are not merging, got %d" % len(recipe.text_filters))

    def test_generate_recipe_with_options(self):
        recipe = generate_recipe(self.doc, self.headers, include_text_filters=True, merge_similar_text_filters=True,
                                 toc_filter_options=self.toc_filter_options, 
                                 text_filter_options=self.text_filter_options)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 3)
        self.assertEqual(len(recipe.text_filters), 3)

    def test_generate_recipe_with_merge_similar_text_filters(self):
        recipe = generate_recipe(self.doc, self.headers, include_text_filters=True, merge_similar_text_filters=True)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 3)
        self.assertEqual(len(recipe.text_filters), 1), "Expected 1 text filter, because we are merging, got %d" % len(recipe.text_filters)

    def test_generate_recipe_with_custom_tolerances(self):
        tolerances = {"font": 0, "bbox": 0}
        recipe = generate_recipe(self.doc, self.headers, tolerances=tolerances, include_text_filters=True, merge_similar_text_filters=True)
        self.assertIsInstance(recipe, Recipe)
        self.assertEqual(len(recipe.toc_filters), 3)
        self.assertEqual(len(recipe.text_filters), 3)

if __name__ == '__main__':
    unittest.main()