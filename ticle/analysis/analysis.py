import numpy as np
from typing import List


def get_significant_periods(data :np.ndarray, n : int) -> List[float]:
    """
    Returns n siginificant periods according to the stellingwerf algorithm.
    :param data: Temporal data containing time and flux
    :return: List of significant periods
    """
    pass


def count_zero_crossings(data : np.ndarray) -> int:
    """
    Counts the zero crossings in a dataset. Zero is in this case the mean of the data. Be aware that a boxcar
    smoothed datset is preferred.
    :param data: Temporal data containing time and flux
    :return: Number of zero crossings.
    """
    pass


def get_periodogram(data : np.ndarray) -> np.ndarray:
    """
    Computes the frequency space of a given dataset.
    :param data: Temporal data containing time and flux
    :return: Dataset containing frequency and power for a given dataset.
    """
    pass


def get_phases(data : np.ndarray,period : float) -> List[np.ndarray]:
    """
    Computes a list of periods according to a given period.
    :param data: Temporal data containing time and flux
    :param period: Period where the data is folded
    :return: List of temporal arrays, normalized from 0 to 1 in time.
    """
    pass


def get_closest_pdm_period(p_guess : float,pdm_list : List[float]) -> float:
    """
    Finds the closes pdm period for a given period.
    :param p_guess: Guess for a given timeseries
    :param pdm_list: List of periods found by pdm
    :return: Closest period for the given guess.
    """
    pass


def get_closest_significant_period(f_data : np.ndarray, p_guess : float) -> float:
    """
    Determines the closest period in frequency space for a given period
    :param f_data: Frequency space of the timeseries
    :param p_guess: Guess of the period
    :return: Closest period in the frequency space.
    """
    pass


def perform_classification(data : np.ndarray) -> bool:
    """
    Performs the classification for a given dataset.
    :param data: Timeseries data. Should be already reduced.
    :return: Boolean value determining the data.
    """
    pass
