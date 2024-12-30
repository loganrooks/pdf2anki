import logging
import time
from typing import List, Set

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

def get_all_indices_within_tolerance(sorted_numbers: List[float], number: float, tolerance: float) -> List[int]:
    """
    Get the indices of all numbers within a tolerance of a given number.

    Args:
        sorted_numbers (List[float]): The list of sorted numbers to search.
        number (float): The number to compare against.
        tolerance (float): The tolerance value.

    Returns:
        List[int]: A list of indices of numbers within the tolerance of the given number.
    """
    # binary search to find the start and end indices
    left = 0
    right = len(sorted_numbers) - 1
    while left < right:
        mid = (left + right) // 2
        if sorted_numbers[mid] < number - tolerance:
            left = mid + 1
        else:
            right = mid
    start = left
    left = 0
    right = len(sorted_numbers) - 1
    while left < right:
        mid = (left + right) // 2
        if sorted_numbers[mid] < number + tolerance:
            left = mid + 1
        else:
            right = mid
    end = left
    return set(range(start, end))
    


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

    group_indices: List[Set[int]] = [[]]

    for number in sorted_numbers:
        group_indices.append(get_all_indices_within_tolerance(sorted_numbers, number, tolerance))

    sorted_group_indices = sorted(group_indices, key=lambda x: len(x), reverse=True)
    grouped_unique_indices = set()
    average_indices = []

    for group in sorted_group_indices:
        unique_group = group - grouped_unique_indices
        if unique_group:
            grouped_unique_indices |= unique_group
            average_indices.append(unique_group)
        else:
            break

    averages = [get_average([sorted_numbers[i] for i in group]) for group in average_indices]

    return averages

