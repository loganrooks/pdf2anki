import logging
import time
from typing import List

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

    groups = [[numbers[0]]]
    averages = [numbers[0]]

    for number in numbers[1:]:
        closest_group_index = min(range(len(averages)), key=lambda i: abs(averages[i] - number))
        if abs(averages[closest_group_index] - number) > tolerance:
            groups.append([number])
            averages.append(number)
        else:
            averages[closest_group_index] += (number - averages[closest_group_index]) / 3
            groups[closest_group_index].append(number)


    return averages

