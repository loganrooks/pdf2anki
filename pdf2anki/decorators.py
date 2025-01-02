from functools import wraps
import logging
import time

import tqdm

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