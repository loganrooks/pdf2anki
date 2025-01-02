import pytest
from pdf2anki.utils import get_average, get_averages, is_valid_dict

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

def test_is_valid_dict_all_valid():
    admissible_types = {
        "name": str,
        "age": int,
        "height": float
    }
    d = {
        "name": "John",
        "age": 30,
        "height": 5.9
    }
    assert is_valid_dict(d, admissible_types) == True

def test_is_valid_dict_invalid_type():
    admissible_types = {
        "name": str,
        "age": int,
        "height": float
    }
    d = {
        "name": "John",
        "age": "thirty",  # Invalid type
        "height": 5.9
    }
    assert is_valid_dict(d, admissible_types) == False

def test_is_valid_dict_missing_key():
    admissible_types = {
        "name": str,
        "age": int,
        "height": float
    }
    d = {
        "name": "John",
        "height": 5.9
    }
    assert is_valid_dict(d, admissible_types) == True  # Missing key is allowed

def test_is_valid_dict_extra_key():
    admissible_types = {
        "name": str,
        "age": int,
        "height": float
    }
    d = {
        "name": "John",
        "age": 30,
        "height": 5.9,
        "weight": 70  # Extra key
    }
    assert is_valid_dict(d, admissible_types) == True  # Extra key is allowed

def test_is_valid_dict_nested_dict():
    admissible_types = {
        "name": str,
        "details": dict
    }
    d = {
        "name": "John",
        "details": {
            "age": 30,
            "height": 5.9
        }
    }
    assert is_valid_dict(d, admissible_types) == True

def test_is_valid_dict_nested_dict_invalid():
    admissible_types = {
        "name": str,
        "details": dict
    }
    d = {
        "name": "John",
        "details": "invalid"  # Invalid type
    }
    assert is_valid_dict(d, admissible_types) == False

if __name__ == "__main__":
    pytest.main()
