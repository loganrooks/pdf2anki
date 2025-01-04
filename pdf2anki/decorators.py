from collections import defaultdict
from functools import wraps
import functools
import logging
import time

import tqdm
from tqdm import tqdm

call_count = defaultdict(int)

def conditional_decorator(decorator, condition):
    """Conditionally apply a decorator."""
    def decorator_wrapper(func):
        if condition:
            return decorator(func)
        return func
    return decorator_wrapper

def count_calls(track_args=None, key_func=None, detect_repeats=True, repeat_threshold=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a key based on the function name and arguments
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                if track_args:
                    tracked_kwargs = {arg: kwargs[arg] for arg in track_args if arg in kwargs}
                    key = (func.__name__, tuple(tracked_kwargs.items()))
                else:
                    key = (func.__name__, args, frozenset(kwargs.items()))
            call_count[key] += 1
            # logging.debug(f"Function {func.__name__} called with key {key} {call_count[key]} times")
            if detect_repeats and call_count[key] > repeat_threshold:
                logging.warning(f"Function {func.__name__} called with key {key} {call_count[key]} times")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_time(func):
    """Decorator to log the time a function takes to run."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def progress_monitor(func):
    """Decorator to display progress for a function with a verbose flag."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        verbose = kwargs.get('verbose', False)
        if verbose:
            # Assuming the function returns an iterable
            iterable = func(*args, **kwargs)
            return list(tqdm(iterable, desc=func.__name__))
        else:
            return func(*args, **kwargs)
    return wrapper