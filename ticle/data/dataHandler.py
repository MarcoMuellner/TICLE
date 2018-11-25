import numpy as np


def load_file(filename : str) ->np.ndarray :
    """
    Loads the data from a given file and returns it as a 2D numpy array
    :param filename: Filename for the file. Needs to be the full path of the file!
    :return: 2D array containing time and flux
    """
    data =  np.loadtxt(filename)
    if len(data) > len(data[0]):
        return data.T
    else:
        return data

def normalizeData(data: np.ndarray) -> np.ndarray:
    """
    This function reduces the dataset to make computations easier. It subtracts the 0 point in the temporal axis and
    removes the mean of the dataset.
    :param data: The dataset, probably read through readData
    :return: A normalized dataset
    """
    data[0] -= data[0][0]

    x = data[0]
    y = data[1]

    for i in [np.inf,-np.inf,np.nan]:
        if i in y:
            n = len(y[y==i])
            print(f"You have {n} {i} in your data. These points will be removed.")

        x = x[y != i]
        y = y[y != i]

    y -= np.mean(y)
    return np.array((x,y))


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

