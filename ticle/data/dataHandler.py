import numpy as np


def load_file(filename : str) ->np.ndarray :
    """
    Loads the data from a given file and returns it as a 2D numpy array
    :param filename: Filename for the file. Needs to be the full path of the file!
    :return: 2D array containing time and flux
    """
    pass


def refine_data(data : np.ndarray) ->np.ndarray:
    """
    Refines the dataset. Performs Interpolation, fixes jumps and gaps, etc.
    :param data: Temporal data containing time and flux
    :return: The reduced dataset, that can be used in further analysis.
    """
    pass


def fix_trends(data : np.ndarray) ->np.ndarray:
    """
    Finds trends in the dataset and tries to fix them, returning a useful dataset.
    :param data: Temporal data containing time and flux
    :return: Untrended dataset.
    """
    pass


def perform_boxcar_smoothing(data : np.ndarray)->np.ndarray:
    """
    Performs a boxcar smoothing on the dataset.
    :param data: Temporal data containing time and flux
    :return: Smoothed dataset.
    """
    pass

