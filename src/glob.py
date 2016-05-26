"""
Setting global variable.
"""


def set_dir():
    """Set directory information."""
    global BASE_PATH
    global DATASET_PATH
    global MODEL_PATH
    global RESULT_PATH
    BASE_PATH = '../'
    DATASET_PATH = BASE_PATH + 'datasets/'
    MODEL_PATH = BASE_PATH + 'models/'
    RESULT_PATH = BASE_PATH + 'results/'
