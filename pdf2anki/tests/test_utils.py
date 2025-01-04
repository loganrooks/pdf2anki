from typing import Optional, Union, Dict
import pdb
import pytest
from pdf2anki.filters import FontFilter, FontFilterVars, FontFilterOptions, BoundingBoxFilter, BoundingBoxFilterVars, BoundingBoxFilterOptions, TextFilter, TextFilterVars, TextFilterOptions, ToCFilterOptions
from pdf2anki.utils import get_average, get_averages, is_valid_arg
from typing import Any, Union, Optional, Dict, List, Set, Tuple, get_origin, get_args

@pytest.mark.parametrize(
    "numbers, expected",
    [
        ([], 0),
        ([10, 20, 30], 20),
        ([1.5, 2.5], 2.0),
    ],
)
def test_get_average_param(numbers, expected):
    assert get_average(numbers) == pytest.approx(expected)

@pytest.mark.parametrize(
    "numbers,tolerance,expected_count",
    [
        ([], 1.0, 0),
        ([10, 11, 12, 13], 5.0, 1),
        ([10, 20, 21, 100, 105], 5.0, 3),
        ([10, 20, 21, 22, 100, 110], 10, 3),
    ],
)
def test_get_averages_param(numbers, tolerance, expected_count):
    result = get_averages(numbers, tolerance)
    assert len(result) == expected_count

@pytest.mark.parametrize(
    "value,type_hint,expected",
    [
        (42, int, True),
        ("42", int, False),
        (None, Optional[int], True),
        ({"key": 2}, Dict[str, int], True),
        ({"key": "val"}, Dict[str, int], False),
        ([1, 2, 3], Union[List[int], Set[int]], True),
        ({"a", "b"}, Union[List[int], Set[int]], True),
        ((1, 2, 3), Tuple[int, ...], True),
        ((1, "two"), Tuple[int, str], True),
        ((1, 2), Tuple[int, str], False),
    ],
)
def test_is_valid_arg_param(value, type_hint, expected):
    assert is_valid_arg(value, type_hint) == expected

def test_get_average_empty_list():
    assert get_average([]) == 0

def test_get_average_basic():
    assert get_average([10, 20, 30]) == 20

def test_get_averages_empty_list():
    assert get_averages([], tolerance=1.0) == []

def test_get_averages_single_group():
    nums = [10, 11, 12, 13]
    result = get_averages(nums, tolerance=5.0)
    # Should return one average of ~11.5
    assert len(result) == 1
    assert abs(result[0] - 11.5) < 1.0

def test_get_averages_multiple_groups():
    nums = [10, 20, 21, 100, 105]
    result = get_averages(nums, tolerance=5.0)
    # Expect ~2 or 3 groups here
    assert len(result) >= 2

def test_get_averages_exact_partition():
    nums = [10, 20, 21, 22, 100, 110]
    result = get_averages(nums, tolerance=10)
    # Groups should be [10], [20, 21, 22], [100, 110]
    assert len(result) == 3
    # Approx checks: first ~10, second ~21, third ~105
    assert abs(result[0] - 10) < 0.5
    assert abs(result[1] - 21) < 1.0
    assert abs(result[2] - 105) < 5.0

@pytest.mark.parametrize(
    "value,hint,expected",
    [
        (42, int, True),
        ("42", int, False),
        (3.14, float, True),
        (True, bool, True),
        ("true", bool, False),
        (None, Optional[int], True),
        ({1, 2, 3}, Set[int], True),
        ({1, "two"}, Set[int], False),
        ([1, 2, 3], List[int], True),
        ([1, "two"], List[int], False),
        ({"key": 10}, Dict[str, int], True),
        ({"key": "val"}, Dict[str, int], False),
        ((1, 2), Tuple[int, int], True),
        ((1, "2"), Tuple[int, int], False),
        ((1, 2, 3), Tuple[int, ...], True),
        ((1, 2, 3), Tuple[str, ...], False),
        ([1, 2, 3], Union[List[int], Set[int]], True),
        ({"a", "b"}, Union[List[int], Set[str]], True),
        (None, Any, True),
        ({"check_font": True, "check_bbox": True, "bbox": BoundingBoxFilterOptions()}, Dict[str, Union[bool, FontFilterOptions, BoundingBoxFilterOptions]], True),
        ({"toc": ToCFilterOptions(), "bbox": BoundingBoxFilterOptions()}, Dict[str, Union[bool, FontFilterOptions, BoundingBoxFilterOptions]], False),
        ({"check_font": True, "check_bbox": True, "bbox": BoundingBoxFilterOptions()}, Dict[str, Union[ToCFilterOptions, FontFilterOptions, BoundingBoxFilterOptions]], False),
        ({"toc": ToCFilterOptions(), "bbox": BoundingBoxFilterOptions(), "font": FontFilterOptions()}, Dict[str, Union[ToCFilterOptions, FontFilterOptions, BoundingBoxFilterOptions]], True)

    ],
)
def test_is_valid_arg_param(value, hint, expected):
    """
    Parametrized tests for is_valid_arg to cover basic, optional, union, set, list, dict, tuple, etc.
    """
    assert is_valid_arg(value, hint) == expected

if __name__ == "__main__":
    pytest.main()
