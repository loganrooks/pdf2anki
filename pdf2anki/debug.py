from pdfminer.layout import LTChar, LTFigure, LTPage

from pdf2anki.elements import CharInfo

def get_first_40_chars_text(lt_page: LTPage) -> str:
    """
    Get the text from the first 40 characters of an LTPage object.
    
    Args:
        lt_page (LTPage): The LTPage object to extract text from.
    
    Returns:
        str: The text from the first 40 characters.
    """
    text = []
    char_count = 0

    for element in lt_page:
        if isinstance(element, LTFigure):
            for sub_element in element:
                if isinstance(sub_element, LTChar):
                    text.append(sub_element.get_text())
                    char_count += 1
                    if char_count >= 40:
                        return ''.join(text)
    return ''.join(text)

# Example usage
# lt_page = ...  # Replace with an actual LTPage object
# print(get_first_40_chars_text(lt_page))

from pdfminer.layout import LTChar, LTFigure

def get_char_attributes_at_index(lt_page: LTPage, index: int) -> dict:
    """
    Get the attributes of the LTChar object at the given index within an LTPage.

    Args:
        lt_page (LTPage): The LTPage object to extract the character from.
        index (int): The index of the character to retrieve attributes for.

    Returns:
        dict: A dictionary of the attributes of the LTChar object.
    """
    char_count = 0

    for element in lt_page:
        if isinstance(element, LTFigure):
            for sub_element in element:
                if isinstance(sub_element, LTChar):
                    if char_count == index:
                        return {attr: getattr(sub_element, attr) for attr in dir(sub_element) if not attr.startswith('__') and not callable(getattr(sub_element, attr))}
                    char_count += 1

    raise IndexError("Character index out of range")

# Example usage
# lt_page = ...  # Replace with an actual LTPage object
# char_attributes = get_char_attributes_at_index(lt_page, 10)
# print(char_attributes)

from typing import List

def filter_chars_by_attribute(chars: List[CharInfo], attribute: str, threshold: float) -> List[CharInfo]:
    """
    Filter CharInfo objects based on a specified float-type attribute and a threshold.

    Args:
        chars (List[CharInfo]): List of CharInfo objects.
        attribute (str): The attribute to filter by (e.g., 'char_width', 'height').
        threshold (float): The threshold value. Only CharInfo objects with the attribute value below this threshold will be included.

    Returns:
        List[CharInfo]: Filtered list of CharInfo objects.
    """
    filtered_chars = [char for char in chars if getattr(char, attribute, float('inf')) < threshold]
    return filtered_chars

# Example usage
# chars = [...]  # Replace with actual list of CharInfo objects
# filtered_chars = filter_chars_by_attribute(chars, 'char_width', 2.5)
# print(filtered_chars)