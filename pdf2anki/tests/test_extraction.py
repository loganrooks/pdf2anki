import math
import pdb
import pickle
from PIL import Image
from typing import Any, Callable, List
import pytest
from pdf2anki.extraction import (
    extract_char_info,
    extract_line_info,
    extract_paragraph_info,
    extract_page_info,
    process_ltpages,
    remove_file_objects,
    remove_lt_images,
    restore_file_objects,
    get_values_from_ltpage,
    save_and_remove_images
)

from pdf2anki.tests.utils import compare_float_sets, create_test_lt_doc, PAGE_HEIGHT, PAGE_WIDTH, insert_headers_at_index
from pdf2anki.utils import concat_bboxes
from pdf2anki.elements import CharInfo, LineInfo, ParagraphInfo, PageInfo, FileObject, FileType
from pdfminer.layout import LTChar, LTPage, LTTextLineHorizontal, LTFigure, LTImage, LTComponent, LTContainer, LTTextContainer, LTLayoutContainer, LTItemT, LAParams
from pdfminer.pdffont import PDFFont
from pdfminer.pdfcolor import PDFColorSpace
from pdfminer.pdfparser import PDFStream, PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import PDFObjRef, DecipherCallable, stream_value
from pdfminer.psparser import literal_name, LIT
import io
import os
from enum import Enum
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# Add mock PDFGraphicState class for testing
class MockPDFGraphicState:
    def __init__(self):
        pass  # Add attributes if necessary

def test_extract_char_info():
    # Initialize required arguments for LTChar
    matrix = (1, 0, 0, 1, 0, 0)  # Identity matrix
    font_descriptor = {"FontName": "TestFont"}
    widths = {ord('A'): 10.0}  # Minimal widths dictionary
    font = PDFFont(descriptor=font_descriptor, widths=widths)  # Properly initialized PDFFont
    font.fontname = "TestFont"  # Manually add the 'fontname' attribute

    fontsize = 10.0
    scaling = 1.0  # Default scaling
    rise = 0.0  # Default rise
    text = 'A'
    textwidth = 1.0
    textdisp = 0.0  # Default text displacement
    ncs = PDFColorSpace(name='DeviceGray', ncomponents=1)  # Properly initialized PDFColorSpace
    graphicstate = MockPDFGraphicState()  # Mock graphic state

    # Create LTChar instance with all required arguments using keyword arguments
    ltchar = LTChar(
        matrix=matrix,
        font=font,
        fontsize=fontsize,
        scaling=scaling,
        rise=rise,
        text=text,
        textwidth=textwidth,
        textdisp=textdisp,
        ncs=ncs,
        graphicstate=graphicstate
    )

    # Extract CharInfo
    char_info = extract_char_info(ltchar)
    
    # Assertions
    assert char_info.text == 'A', "Char text does not match"
    assert char_info.bbox == (0.0, 0.0, 10.0, 10.0), "Char bbox does not match"
    assert char_info.size == 10.0, "Char size does not match"
    assert char_info.font == 'TestFont', "Char font does not match"
    assert char_info.color == 'DeviceGray', "Char color does not match"
    assert char_info.height == 10.0, "Char height does not match"
    assert char_info.width == 10.0, "Char width does not match"

def test_extract_line_info():
    # Mock CharInfo objects
    char1 = CharInfo(text='H', bbox=(0, 0, 10, 10), size=12.0, font='FontA', color='black', height=10.0, width=10.0)
    char2 = CharInfo(text='i', bbox=(10, 0, 18, 10), size=12.0, font='FontA', color='black', height=10.0, width=8.0)
    line = [char1, char2]

    # Extract LineInfo
    line_info = extract_line_info(line)
    
    # Assertions
    assert line_info.text == 'Hi', "Line text does not match"
    assert line_info.bbox == (0, 0, 18, 10), "Line bbox does not match"
    assert line_info.font_size == 12.0, "Line font size does not match"
    assert line_info.fonts == frozenset({'FontA'}), "Line fonts do not match"
    assert line_info.colors == frozenset({'black'}), "Line colors do not match"
    assert line_info.char_width == 9.0, "Line char width does not match"
    assert line_info.char_height == 10.0, "Line char height does not match"
    assert not line_info.split_end_word, "Line split_end_word should be False"

def test_concat_bboxes():
    bboxes = [(0, 0, 10, 10), (10, 0, 20, 10), (20, 0, 30, 10)]
    concatenated = concat_bboxes(bboxes)
    assert concatenated == (0, 0, 30, 10), "Concatenated bbox does not match"

def test_extract_paragraph_info():
    # Mock LineInfo objects
    line1 = LineInfo(
        text="This is a test. ",
        chars=(),
        bbox=(0, 0, 100, 10),
        font_size=12.0,
        fonts=frozenset({'FontA'}),
        colors=frozenset({'black'}),
        char_width=10.0,
        char_height=10.0,
        split_end_word=False
    )
    line2 = LineInfo(
        text="This is a continuation. ",
        chars=(),
        bbox=(0, 10, 120, 20),
        font_size=12.0,
        fonts=frozenset({'FontA'}),
        colors=frozenset({'black'}),
        char_width=10.0,
        char_height=10.0,
        split_end_word=False
    )
    paragraph = [line1, line2]

    # Extract ParagraphInfo
    paragraph_info = extract_paragraph_info(paragraph, pagenum=1, indent_factor=3.0)
    
    # Assertions
    assert paragraph_info.text == "This is a test. This is a continuation. ", "Paragraph text does not match"
    assert paragraph_info.bbox == (0, 0, 120, 20), "Paragraph bbox does not match"
    assert paragraph_info.fonts == frozenset({'FontA'}), "Paragraph fonts do not match"
    assert paragraph_info.colors == frozenset({'black'}), "Paragraph colors do not match"
    assert paragraph_info.char_width == 10.0, "Paragraph char width does not match"
    assert paragraph_info.font_size == 12.0, "Paragraph font size does not match"
    assert not paragraph_info.split_end_line, "Paragraph split_end_line should be False"
    assert not paragraph_info.is_indented, "Paragraph is_indented should be False"

def test_extract_page_info():
    # Mock ParagraphInfo objects
    paragraph1 = ParagraphInfo(
        pagenum=1,
        text="First paragraph.",
        lines=(),
        bbox=(0, 100, 100, 150),
        fonts=frozenset({'FontA'}),
        colors=frozenset({'black'}),
        char_width=10.0,
        font_size=12.0,
        split_end_line=False,
        is_indented=False
    )
    paragraph2 = ParagraphInfo(
        pagenum=1,
        text="Second paragraph.",
        lines=(),
        bbox=(0, 50, 100, 100),
        fonts=frozenset({'FontA'}),
        colors=frozenset({'black'}),
        char_width=10.0,
        font_size=12.0,
        split_end_line=False,
        is_indented=False
    )
    page = [paragraph1, paragraph2]

    # Extract PageInfo
    page_info = extract_page_info(page, font_size_grouping_threshold=0.1)
    
    # Assertions
    assert page_info.text == "First paragraph.\n\nSecond paragraph.", "Page text does not match"
    assert page_info.bbox == (0, 50, 100, 150), "Page bbox does not match"
    assert page_info.fonts == frozenset({'FontA'}), "Page fonts do not match"
    assert page_info.font_sizes == frozenset([12.0]), "Page font sizes do not match"
    assert page_info.char_widths == frozenset([10.0]), "Page char widths do not match"
    assert page_info.colors == frozenset({'black'}), "Page colors do not match"
    assert page_info.paragraphs == tuple(page), "Page paragraphs do not match"
    assert not page_info.split_end_paragraph, "Page split_end_paragraph should be False"
    assert not page_info.starts_with_indent, "Page starts_with_indent should be False"


def test_extract_page_info_two_fonts():
    # Mock ParagraphInfo objects
    paragraph1 = ParagraphInfo(
        pagenum=1,
        text="First paragraph.",
        lines=(),
        bbox=(0, 100, 100, 150),
        fonts=frozenset({'FontA'}),
        colors=frozenset({'black'}),
        char_width=12.0,
        font_size=14.0,
        split_end_line=False,
        is_indented=False
    )
    paragraph2 = ParagraphInfo(
        pagenum=1,
        text="Second paragraph.",
        lines=(),
        bbox=(0, 50, 100, 100),
        fonts=frozenset({'FontB'}),
        colors=frozenset({'red'}),
        char_width=10.0,
        font_size=12.0,
        split_end_line=False,
        is_indented=False
    )
    paragraph3 = ParagraphInfo(
        pagenum=1,
        text="Second paragraph.",
        lines=(),
        bbox=(0, 0, 100, 50),
        fonts=frozenset({'FontB'}),
        colors=frozenset({'red'}),
        char_width=10.0,
        font_size=12.0,
        split_end_line=False,
        is_indented=False
    )
    page = [paragraph1, paragraph2, paragraph3]

    # Extract PageInfo
    page_info = extract_page_info(page, font_size_grouping_threshold=0.1)
    assert page_info.fonts == frozenset({'FontA', 'FontB'}), "Page fonts do not match"
    assert page_info.font_sizes == frozenset([12.0, 14.0]), "Page font sizes do not match"
    assert page_info.char_widths == frozenset([10.0, 12.0]), "Page char widths do not match"
    assert page_info.colors == frozenset({'black', 'red'}), "Page colors do not match"

def test_process_ltpages_no_type_error():

    text_blocks = [
                # Page 1 has three paragraphs
                "Paragraph 1: This is the first paragraph on page 1. \nLine 2.\n\n"
                "Paragraph 2: This is the second paragraph on page 1. \nLine 2.\n\n"
                "Paragraph 3: This is the third paragraph on page 1. \nLine 2",
                
                # Page 2 has three paragraphs
                "Paragraph 1: This is the first paragraph on page 2. \nLine 2\n\n"
                "Paragraph 2: This is the second paragraph on page 2. \nLine 2\n\n"
                "Paragraph 3: This is the third paragraph on page 2. \nLine 2",
                
                # Page 3 has three paragraphs
                "Paragraph 1: This is the first paragraph on page 3. \nLine 2\n\n"
                "Paragraph 2: This is the second paragraph on page 3. \nLine 2\n\n"
                "Paragraph 3: This is the third paragraph on page 3. \nLine 2"
            ]
        
    headers = [
            ("Header for Page 1", 0),  # Insert at first paragraph on Page 1
            ("Header for Page 2", 1),  # Insert at second paragraph on Page 2
            ("Header for Page 3", 2)   # Insert at third paragraph on Page 3
        ]

    doc = create_test_lt_doc(text_blocks, headers, text_size=12.0, header_size=16.0, font_name="Arial")
    text_blocks_with_headers = insert_headers_at_index(text_blocks, headers)
    text_blocks_with_headers = ['\n\n'.join([paragraph for paragraph, _ in page]) for page in text_blocks_with_headers]

    params = LAParams(line_overlap=0.7, char_margin=3.0, line_margin=0.5)

    # Run process_ltpages
    try:
        processed_pages = process_ltpages(doc, char_margin_factor=params.char_margin, line_margin_factor=params.line_margin)
        processed_pages[0].update_pagenum(1)
    except TypeError as e:
        pytest.fail(f"TypeError was raised: {e}")
    comparison_text = text_blocks_with_headers[0].replace(' \n', ' ')
    
    # Assertions
    assert len(processed_pages) == 3, "Processed pages count mismatch."
    processed_page = processed_pages[0]
    assert processed_page.text == comparison_text, "Page text does not match."
    comparison_bbox = (0, 534.0, 274.32, 650.0)
    for i, coord in enumerate(processed_page.bbox):
        assert math.isclose(coord, comparison_bbox[i], rel_tol=1e-6), f"Page bbox does not match at index {i}"
    assert processed_page.fonts == frozenset({'Arial'}), "Page fonts do not match."
    assert processed_page.font_sizes == frozenset([12.0, 16.0]), "Page font sizes do not match."
    assert compare_float_sets(processed_page.char_widths, frozenset([4.75, 6.59]), abs_tol=1e-1), "Page char widths do not match."
    assert processed_page.colors == frozenset({'DeviceGray'}), "Page colors do not match."
    # assert processed_page.paragraphs[0] == (paragraph_info,), "Page paragraphs do not match."
    assert processed_page.split_end_paragraph, "Page split_end_paragraph should be True."
    assert not processed_page.starts_with_indent, "Page starts_with_indent should be False."

class MockLTContainer(LTContainer):
    def __init__(self, bbox):
        LTContainer.__init__(self, bbox)
    
    def __iter__(self):
        return iter(self._objs)
    
    def add_lt_image(self, bbox, file_name, create_pdf=True):
        j = len([obj for obj in self._objs if isinstance(obj, LTImage)])
        img = MockLTImage(bbox, file_name, create_pdf=create_pdf, i=j)
        self._objs.append(img)
        return self
    
    def add_lt_figure(self, bbox, matrix):
        j = len([obj for obj in self._objs if isinstance(obj, LTFigure)])
        fig = MockLTFigure(bbox, matrix, i=j)
        self._objs.append(fig)
        return self

    def add_lt_char(self, font, fontsize, text, textwidth, bbox, matrix=(1, 0, 0, 1, 0, 0), scaling=1.0, rise=0.0, textdisp=0.0, ncs=None, graphicstate=None):
        char = LTChar(matrix=matrix, font=font, fontsize=fontsize, scaling=scaling, rise=0.0, text=text, textwidth=textwidth, textdisp=0.0, ncs=None, graphicstate=None)
        char.set_bbox(bbox)
        self.add(char)
        return self

    def add_lt_textline(self, objs: List[LTChar]):
        line = LTTextLineHorizontal(word_margin=0.5)
        for obj in objs:
            line.add(obj)
        self._objs.append(line)
        return self

class StreamType(Enum):
    IMAGE = 0
    TEXT = 1


class MockPDFParser(PDFParser):
    def __init__(self, file_name: str):
        fp = open(file_name, 'rb')
        super().__init__(fp)

class MockPDFDocument(PDFDocument):
    def __init__(self, file_name: str):
        parser = MockPDFParser(file_name)
        super().__init__(parser)

class MockPDFObjRef(PDFObjRef):
    def __init__(self, file_name, i=0):
        doc = MockPDFDocument(file_name)
        super().__init__(doc, i, None)

    
class MockPDFStream(PDFStream):
    def __init__(self, file_name, create_pdf: bool = True, type: StreamType = StreamType.IMAGE, i=0):
        
        pdf_file_name = os.path.splitext(file_name)[0] + ".pdf"
        if type == StreamType.IMAGE:
            with Image.open(file_name, 'r') as img:
                width, height = img.size
                bits_per_component = 8 if img.mode in ["RGB", "L"] else 1  # Adjust based on mode
                color_space = "DeviceRGB" if img.mode in ["RGB", "L"] else "DeviceGray"
                length = len(img.tobytes())
                rawdata = img.tobytes()


                decode_params = {
                    "BitsPerComponent": bits_per_component,
                    "Colors": 1 if color_space == "DeviceGray" else 3,
                    "Columns": width,
                    "Predictor": 15  # Example value; adjust based on specific encoding
                }
                attrs = {
                        "Type": LIT("XObject"),
                        "Subtype": LIT("Image"),
                        "Width": width,
                        "Height": height,
                        "ColorSpace": [LIT("ICCBased"), MockPDFObjRef(pdf_file_name, i)] if create_pdf else [LIT(color_space)],
                        "BitsPerComponent": bits_per_component,
                        "DecodeParms": decode_params,
                        "Filter": LIT("FlateDecode"),  # Example value; adjust based on specific encoding
                        "Length": length
                    }
        super().__init__(attrs, rawdata)

class MockLTImage(LTImage):
    def __init__(self, bbox, file_name, i=0, create_pdf=True):
        stream = MockPDFStream(file_name, create_pdf, type=StreamType.IMAGE)
        stream.decode()
        super().__init__(f"mock_img_{i}", stream, bbox)

class MockLTFigure(LTFigure, MockLTContainer):
    def __init__(self, bbox, matrix, i=0):
        name = f"mock_fig_{i}"
        LTFigure.__init__(self, name, bbox, matrix)
        MockLTContainer.__init__(self, bbox)

class MockLTPage(LTPage, MockLTContainer):

    def __init__(self, pageid, bbox):
        MockLTContainer.__init__(self, bbox)
        LTPage.__init__(self, pageid, bbox)
        

def create_pdf(file_name: str, size: tuple, color: str) -> str:
    """
    Create a PDF with the specified size and color, and save it to the given file name.
    """
    c = canvas.Canvas(file_name, pagesize=(PAGE_WIDTH, PAGE_HEIGHT))
    c.setFillColor(color)
    c.rect(0, 0, size[0], size[1], fill=1)
    c.save()
    return file_name

def create_image(file_name: str, size: tuple, color: str) -> str:
    """
    Create an image with the specified size and color, and save it to the given file name.
    """
    img = Image.new("RGB", size, color)
    img.save(file_name)
    return file_name

test_files_dir = os.path.join(os.path.dirname(__file__), 'test_files')
os.makedirs(test_files_dir, exist_ok=True)

@pytest.mark.parametrize("size, color, expected_removed_count", [
    ((10, 10), "red", 1),
    ((20, 20), "blue",  1)
])
def test_remove_file_objects_with_different_pdfs(size: tuple, color: str, expected_removed_count: int):
    """
    Test remove_file_objects with parametrized fake PDFs.
    """
    fake_file_name = os.path.join(test_files_dir, f"fake_file_{size[0]}x{size[1]}_{color}.pdf")
    fake_img_name = os.path.splitext(fake_file_name)[0] + ".png"
    # Create a PDF with the specified size and color
    if not os.path.exists(fake_file_name):
        create_pdf(fake_file_name, size, color)
    
    if not os.path.exists(fake_img_name):
        create_image(fake_img_name, size, color)

    lt_page = MockLTPage(pageid=3, bbox=(0, 0, PAGE_WIDTH, PAGE_HEIGHT))
    # Inject the file object into lt_page._objs
    lt_page.add_lt_figure((0, 0, 100, 100), (1, 0, 0, 1, 0, 0))

    kwargs = {"bbox": (0, 0, size[0], size[1]), "file_name": fake_img_name}    
    assert len(lt_page._objs) == 1 and isinstance(lt_page._objs[0], MockLTFigure), "LTPage should contain a single LTFigure object."

    for _ in range(expected_removed_count):
        lt_page._objs[0].add_lt_image(**kwargs)
    
    # pdb.set_trace()

    removed = remove_file_objects(lt_page)
    assert len(removed) == expected_removed_count, f"Expected {expected_removed_count} removed file objects, got {len(removed)}."
    assert isinstance(removed[0], FileObject), "Removed object should be a FileObject."

@pytest.mark.parametrize("num_files, size, color", [
    (1, (10, 10), "red"),
    (3, (20, 20), "blue"),
    (5, (30, 30), "green")
])
def test_remove_file_objects_multiple_files(num_files, size, color):
    """
    Test remove_file_objects with multiple fake file objects.
    """
    lt_page = MockLTPage(pageid=3, bbox=(0, 0, PAGE_WIDTH, PAGE_HEIGHT))
    lt_page.add_lt_figure((0, 0, 50, 50), (1, 0, 0, 1, 0, 0))

    assert len(lt_page._objs) == 1 and isinstance(lt_page._objs[0], MockLTFigure), "LTPage should contain a single LTFigure object."

    for i in range(num_files):
        fake_file_name = os.path.join(test_files_dir, f"fake_pdf_{size[0]}x{size[1]}_{color}_{i}.pdf")
        fake_img_name = os.path.splitext(fake_file_name)[0] + ".png"
        if not os.path.exists(fake_file_name):
            create_pdf(fake_file_name, size, color)
        if not os.path.exists(fake_img_name):
            create_image(fake_img_name, size, color)
        lt_page._objs[0].add_lt_image(bbox=(0, 0, size[0], size[1]), file_name=fake_img_name)

    removed = remove_file_objects(lt_page)
    assert len(removed) == num_files, f"Expected {num_files} removed file objects, got {len(removed)}."

@pytest.mark.parametrize("size, color, expected_removed_count", [
    ((10, 10), "red", 2),
    ((20, 20), "blue",  2)
])
def test_remove_and_restore_file_objects(size, color, expected_removed_count):
    """
    Test remove_file_objects followed by restore_file_objects with multiple fake files.
    """
    fake_file_name = os.path.join(test_files_dir, f"fake_pdf_{size[0]}x{size[1]}_{color}.pdf")
    fake_img_name = os.path.splitext(fake_file_name)[0] + ".png"

    # Create a PDF with the specified size and color
    if not os.path.exists(fake_file_name):
        create_pdf(fake_file_name, size, color)
    if not os.path.exists(fake_img_name):
        create_image(fake_img_name, size, color)

    lt_page = MockLTPage(pageid=3, bbox=(0, 0, PAGE_WIDTH, PAGE_HEIGHT))
    # Inject the file object into lt_page._objs
    lt_page.add_lt_figure((0, 0, 100, 100), (1, 0, 0, 1, 0, 0))

    kwargs = {"bbox": (0, 0, size[0], size[1]), "file_name": fake_img_name}    
    assert len(lt_page._objs) == 1 and isinstance(lt_page._objs[0], MockLTFigure), "LTPage should contain a single LTFigure object."

    for _ in range(expected_removed_count):
        lt_page._objs[0].add_lt_image(**kwargs)

    # Remove
    removed = remove_file_objects(lt_page)
    assert len(removed) == expected_removed_count, f"Expected {expected_removed_count} removed file objects, got {len(removed)}."

    # Restore
    restore_file_objects(removed, lt_page)
    restored = get_values_from_ltpage(lt_page, file_objects=removed)
    assert len(restored) == expected_removed_count, f"Expected {expected_removed_count} restored file objects, got {len(lt_page._objs)}."
    
    for lt_image in lt_page._objs[0]:
        assert isinstance(lt_image, LTImage), "Each element of the LTFigure should be an LTImage."
        pdf_obj_ref = lt_image.colorspace[1]
        assert isinstance(pdf_obj_ref, PDFObjRef), "Each colorspace should have a PDFObjRef."
        parser = pdf_obj_ref.doc._parser
        assert isinstance(parser, MockPDFParser), "Each PDFObjRef should have a MockPDFParser."
        restored_item = parser.fp
        assert restored_item is not None, "Each restored file should not be None."

    second_removal = remove_file_objects(lt_page)
    assert second_removal == removed, "Second removal should return the same list of removed file objects."




@pytest.fixture
def mock_ltpage_with_figure_and_images():

    image1_path = os.path.join(test_files_dir, "image1.png")
    image2_path = os.path.join(test_files_dir, "image2.png")

    if not os.path.exists(image1_path):
        create_image(image1_path, (10, 10), "red")
    if not os.path.exists(image2_path):
        create_image(image2_path, (20, 20), "blue")

    lt_page = MockLTPage(pageid=1, bbox=(0, 0, 100, 100))
    lt_page.add_lt_figure(bbox=(0, 0, 50, 50), matrix=(1, 0, 0, 1, 0, 0))
    assert isinstance(lt_page._objs[0], MockLTFigure), "LTPage should contain a single LTFigure object."
    lt_page._objs[0].add_lt_image(bbox=(0, 0, 10, 10), file_name=image1_path, create_pdf=False)
    lt_page._objs[0].add_lt_image(bbox=(10, 10, 30, 30), file_name=image2_path, create_pdf=False)
    return lt_page

def test_remove_lt_images(mock_ltpage_with_figure_and_images: Callable[[], LTPage]):
    lt_page = mock_ltpage_with_figure_and_images
    assert len(lt_page._objs[0]._objs) == 2, "There should be 2 images in the figure."
    
    cleaned_page = remove_lt_images(lt_page)
    assert len(cleaned_page._objs[0]._objs) == 0, "All images should be removed from the figure."

def test_save_and_remove_images(tmp_path, mock_ltpage_with_figure_and_images: Callable[[], LTPage]):
    lt_page = mock_ltpage_with_figure_and_images
    assert len(lt_page._objs[0]._objs) == 2, "There should be 2 images in the figure."
    
    filepath = tmp_path / "cleaned_pages.pkl"
    save_and_remove_images([lt_page], str(filepath))
    
    with open(filepath, "rb") as f:
        cleaned_pages = pickle.load(f)
    
    assert len(cleaned_pages) == 1, "There should be 1 cleaned page."
    assert len(cleaned_pages[0]._objs[0]._objs) == 0, "All images should be removed from the figure in the saved page."



