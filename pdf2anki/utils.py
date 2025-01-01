import logging
import time
from typing import List, Set, Tuple

# from pdf2anki.tests.test_utils import test_get_averages_exact_partition, test_get_averages_multiple_groups

def log_time(func):
    """Decorator to log the time a function takes to run."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

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
        bbox1 (tuple): Bounding box 1.
        bbox2 (tuple): Bounding box 2.
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

def main():
    # test_get_averages_multiple_groups()
    # test_get_averages_exact_partition()
    pass

if __name__ == "__main__":
    main()
