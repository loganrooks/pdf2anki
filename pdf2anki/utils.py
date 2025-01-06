from typing import Any, Dict, List, Required, Set, Tuple, Union, Optional, get_args, get_origin
import pdb

from pdf2anki.elements import PageInfo




def is_valid_arg(value: Any, hint: Any) -> bool:
    """
    Checks whether 'value' is a valid instance of 'hint', where 'hint'
    can be a complex type annotation involving Optional, Union, dict,
    list, set, tuple, and a special Required[...] for ensuring at least
    one required type is present in a dictionary's values.

    Example:
    Optional[dict[str, Union[dict[str, Union[str, float, bool]],
                               FontFilterOptions,
                               BoundingBoxFilterOptions,
                               Required[ToCFilterOptions]]]]
    """

    # [CHANGED] Handle Any explicitly. If hint is Any, everything is valid.
    if hint is Any:
        return True

    origin = get_origin(hint)
    args = get_args(hint)

    # 1) Handle Optional[...] (Union[..., None])
    if origin is Union and type(None) in args:
        non_none_args = [a for a in args if a is not type(None)]
        if value is None:
            return True
        return any(is_valid_arg(value, a) for a in non_none_args)

    # 2) Handle general Union[...] (no Optional)
    if origin is Union:
        return any(is_valid_arg(value, a) for a in args)

    # 3) Handle Required[...] wrapper
    if origin is Required:
        required_type = args[0] if args else Any
        return is_valid_arg(value, required_type)

    # 4a) Handle list[...] (e.g. list[int])
    if origin is list:
        if not isinstance(value, list):
            return False
        if len(args)==1 or (len(args) == 2 and args[1] is Ellipsis):
            (item_type, ) = args
            return all(is_valid_arg(item, item_type) for item in value)
        # If it’s a fixed-length list (Tuple[T1, T2, ...]) match by position
        if len(value) != len(args):
            return False
        return all(is_valid_arg(item, t) for item, t in zip(value, args))

    # 4b) Handle set[...] (e.g. set[str])
    if origin is set:
        if not isinstance(value, set):
            return False
        if len(args)==1 or (len(args) == 2 and args[1] is Ellipsis):
            (item_type, ) = args
            return all(is_valid_arg(item, item_type) for item in value)
        # If it’s a fixed-length tuple (Tuple[T1, T2, ...]) match by position
        if len(value) != len(args):
            return False
        return all(is_valid_arg(item, t) for item, t in zip(value, args))

    # 4c) Handle tuple[...] (can be Tuple[T, ...] or fixed-length)
    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        # If it’s a variable-length tuple (one type param), check each item
        # pdb.set_trace()
        if len(args)==1 or (len(args) == 2 and args[1] is Ellipsis):
            (item_type, _) = args
            return all(is_valid_arg(item, item_type) for item in value)
        # If it’s a fixed-length tuple (Tuple[T1, T2, ...]) match by position
        if len(value) != len(args):
            return False
        return all(is_valid_arg(item, t) for item, t in zip(value, args))

    # 4d) Handle dict[...] (likely includes TypedDict at runtime)
    if origin is dict:
        if not isinstance(value, dict):
            return False
        dict_key_type, dict_value_type = args  # Renamed for clarity

        # Check that all keys match dict_key_type
        if not all(isinstance(k, dict_key_type) for k in value.keys()):
            return False

        val_origin = get_origin(dict_value_type)
        val_args = get_args(dict_value_type)

        # Collect Required subtypes if present
        required_subtypes = []
        if val_origin is Union:
            for subhint in val_args:
                if get_origin(subhint) is Required:
                    required_subtypes.append(get_args(subhint)[0])
        elif val_origin is Required:
            required_subtypes.append(val_args[0])

        valid_so_far = True
        found_required_types = {req: False for req in required_subtypes}

        for v in value.values():
            if not is_valid_arg(v, dict_value_type):
                valid_so_far = False
                break
            # Mark any Required[...] type found
            for req_type in found_required_types:
                if isinstance(v, req_type):
                    found_required_types[req_type] = True

        if valid_so_far:
            return all(found_required_types[req] for req in found_required_types)
        else:
            return False

    # 5) Basic check for normal types (str, int, bool, classes, etc.)
    if isinstance(hint, type):
        return isinstance(value, hint)

    # 6) Fallback for unhandled cases
    return isinstance(value, hint)

def get_average(numbers):
    """
    Calculate the average of a list of numbers.

    :param numbers: List of numbers
    :return: Average of the numbers
    """
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


def get_all_numbers_within_tolerance(numbers: List[float], number: float, tolerance: float) -> List[float]:
    """
    Get all numbers within a tolerance of a given number.

    Args:
        numbers (List[float]): The list of numbers to search.
        number (float): The number to compare against.
        tolerance (float): The tolerance value.

    Returns:
        List[float]: A list of numbers within the tolerance of the given number.
    """
    return [num for num in numbers if abs(num - number) <= tolerance]

def get_average_L1_distances_from_number(numbers: List[float], number: float) -> float:
    """
    Calculate the average L1 distance of a list of numbers from a given number.

    Args:
        numbers (List[float]): The list of numbers.
        number (float): The number to compare against.

    Returns:
        float: The average L1 distance of the numbers from the given number.
    """
    if numbers:
        return sum(abs(num - number) for num in numbers) / len(numbers)
    else:
        return 0

def get_all_indices_within_tolerance(sorted_numbers: List[float], number: float, tolerance: float) -> List[int]:
    """
    Get the indices of all numbers within a tolerance of a given number and the average L1 distance.

    Args:
        sorted_numbers (List[float]): The list of sorted numbers to search.
        number (float): The number to compare against.
        tolerance (float): The tolerance value.

    Returns:
        Tuple[float, List[int]]: A tuple containing the average distance and the list of indices of numbers within the tolerance of the given number.
    """
    # binary search to find the start and end indices
    left = 0
    right = len(sorted_numbers) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_numbers[mid] < number - tolerance:
            left = mid + 1
        else:
            right = mid - 1
    start = left
    left = 0
    right = len(sorted_numbers) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_numbers[mid] <= number + tolerance:
            left = mid + 1
        else:
            right = mid - 1
    end = left
    indices = set(range(start, end))
    avg_l1_distance = get_average_L1_distances_from_number([sorted_numbers[i] for i in indices if sorted_numbers[i] != number], number)
    return avg_l1_distance, indices

def remove_duplicate_sets(data: List[Tuple[float, Set[int]]]) -> List[Tuple[float, Set[int]]]:
    """
    Remove elements with duplicate sets from a list of tuples containing a float and a set.

    Args:
        data (List[Tuple[float, Set[int]]]): The list of tuples to process.

    Returns:
        List[Tuple[float, Set[int]]]: A list with duplicate sets removed.
    """
    seen_sets = {}
    unique_data = []

    for item in data:
        value, group_set = item
        frozen_set = frozenset(group_set)  # Convert set to frozenset to make it hashable
        if frozen_set not in seen_sets:
            seen_sets[frozen_set] = value
            unique_data.append(item)

    return unique_data

    

def get_averages(numbers: List[float], tolerance: float) -> List[List[float]]:
    """
    Group numbers based on a tolerance value and return their averages.

    Args:
        numbers (List[float]): The list of numbers to group.
        tolerance (float): The tolerance value for grouping.

    Returns:
        List[List[float]]: A list of groups, where each group is a list with two elements:
                           the average of the group and the members of the group.
    """
    if not numbers:
        return []
    
    sorted_numbers = sorted(numbers)

    # averages = [sorted_numbers[0]]
    # groups = [[sorted_numbers[0]]]

    group_indices: List[Set[int]] = []

    for number in sorted_numbers:
        group_indices.append(get_all_indices_within_tolerance(sorted_numbers, number, tolerance))
    
    l1_distance_factor = 0.5
    group_size_factor = 0.5
    
    sorted_group_indices = sorted(group_indices, key=lambda x: group_size_factor*len(x[1]) - l1_distance_factor*x[0] , reverse=True)
    grouped_unique_indices = set()
    average_indices = []
    
    unique_sorted_group_indices = remove_duplicate_sets(sorted_group_indices)

    for _, group in unique_sorted_group_indices:
        unique_group_indices = group - grouped_unique_indices
        if unique_group_indices:
            grouped_unique_indices |= unique_group_indices
            average_indices.append(unique_group_indices)

    averages = sorted([get_average([sorted_numbers[i] for i in group]) for group in average_indices])

    return averages

def concat_bboxes(bboxes: List[Tuple[float]]) -> Tuple[float]:
    """
    Concatenate a list of bounding boxes into a single bounding box.

    Args:
        bboxes (list): A list of bounding boxes.

    Returns:
        tuple: A single bounding box that encompasses all the input bounding boxes.
    """
    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[2] for bbox in bboxes)
    y1 = max(bbox[3] for bbox in bboxes)
    return (x0, y0, x1, y1)

def contained_in_bbox(bbox1: Tuple[float], bbox2: Tuple[float], bbox_overlap: float = 1.0) -> bool:
    """
    Check if bbox1 is contained in bbox2 based on the overlap percentage.

    Args:
        bbox1 (tuple): Bounding box 1. Given as (left, bottom, right, top). Where bottom < top and left < right.
        bbox2 (tuple): Bounding box 2. Given as (left, bottom, right, top). Where bottom < top and left < right.
        bbox_overlap (float): Overlap percentage of bbox1's area that must be in bbox2.

    Returns:
        bool: True if bbox1 is contained in bbox2, False otherwise.
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    # Calculate the area of bbox1
    area1 = (x2 - x1) * (y2 - y1)

    # Calculate the intersection area
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        intersection_area = 0

    return intersection_area >= bbox_overlap * area1

def get_y_overlap(bbox1: Tuple[float], bbox2: Tuple[float]) -> float:
    """
    Calculate the vertical overlap between two bounding boxes.

    Args:
        bbox1 (tuple): Bounding box 1.
        bbox2 (tuple): Bounding box 2.

    Returns:
        float: The vertical overlap between the two bounding boxes.
    """
    y1, y2 = bbox1[1], bbox1[3]
    y3, y4 = bbox2[1], bbox2[3]
    return min(y2, y4) - max(y1, y3)

def clean_text(text: str) -> str:
    """
    Clean up text by removing leading/trailing whitespaces and converting to lowercase.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    return ' '.join(text.split()).replace('“', '"').replace('”', '"').replace("’", "'").lstrip()

def get_text_index_from_vpos(start_vpos: float, 
                             page: PageInfo) -> int:
    """
    Get the index of the first character below the given vertical position in the text string.
    
    Args:
        start_vpos: Vertical position to start searching from.
        page: PageInfo object representing the page.

    Returns:
        int: Index of the first character below the given vertical position.
    """
    start_index = 0
    for paragraph in page.paragraphs:
        if paragraph.bbox[1] < start_vpos:
            for line in paragraph.lines:
                if line.bbox[1] >= start_vpos:
                    start_index += len(line.text)
                else:
                    break
            break
        start_index += len(paragraph.text + "\n\n")
    return start_index

def main():
    # test_get_averages_multiple_groups()
    # test_get_averages_exact_partition()
    pass

if __name__ == "__main__":
    main()
