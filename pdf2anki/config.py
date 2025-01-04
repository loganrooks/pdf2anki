DEBUG = False
VERBOSE = False

def set_debug(value: bool, verbose: bool = False):
    global DEBUG, VERBOSE
    DEBUG = value
    VERBOSE = verbose