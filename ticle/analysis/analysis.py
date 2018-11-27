import numpy as np
from typing import List,Union,Tuple
from scipy.fftpack import fft,fftfreq
from scipy.signal import find_peaks
from scipy.stats import binned_statistic
from astropy.convolution import convolve, Box1DKernel
from astropy.stats import LombScargle

from ticle.analysis.pdm import stellingwerf_pdm


def get_significant_periods(data :np.ndarray, n : int,fullData = False) -> List[float]:
    """
    Returns n siginificant periods according to the stellingwerf algorithm.

    Be careful using this: If the resolution of the dataset is too small, this will not return appropriate periods!
    :param data: Temporal data containing time and flux
    :return: List of significant periods
    """
    sigma = np.zeros(len(data[0])) #this is the error for the lightcurve.
    pdm = stellingwerf_pdm(data[0], data[1], sigma, nbestpeaks=n)
    if fullData:
        return pdm
    else:
        return pdm['nbestperiods']


def count_zero_crossings(data : np.ndarray,getIndices : bool = False) -> Union[int,Tuple[np.ndarray,int]]:
    """
    Counts the zero crossings in a dataset. Zero is in this case the mean of the data. Be aware that a boxcar
    smoothed datset is preferred.
    :param data: Temporal data containing time and flux
    :return: Number of zero crossings.
    """
    y = data[1] - np.mean(data[1])
    zc = ((y[:-1] * y[1:]) < 0).sum()
    if getIndices:
        ind = np.where((y[:-1] * y[1:])<0)
        return zc,ind
    else:
        return zc


def find_most_common_diff(t_data: np.ndarray) -> float:
    """
    Finds the most common time difference between two data points.
    :param t_data: 
    :return:
    """
    realDiffX = t_data[1:len(t_data)] - t_data[0:len(t_data) - 1]
    realDiffX = realDiffX[realDiffX!=0]
    (values, counts) = np.unique(realDiffX, return_counts=True)
    mostCommon = values[np.argmax(counts)]
    return mostCommon


def get_periodogram(data : np.ndarray) -> np.ndarray:
    """
    Computes the frequency space of a given dataset.
    :param data: Temporal data containing time and flux
    :return: Dataset containing frequency and power for a given dataset.
    """
    f_s = find_most_common_diff(data[0])
    f_x = fftfreq(len(data[0])) * 1/f_s
    f_y = fft(data[1])[f_x > 0]
    f_x = f_x[f_x > 0]

    return np.array((f_x,np.abs(f_y)))

def nyquist_f(data: np.ndarray) -> float:
    """
    Computes Nyquist frequency.
    :param data: Dataset, in time domain.
    :return: Nyquist frequency of dataset
    """
    return float(1 / (2 * find_most_common_diff(data[0])))

def get_lomb_scargle(data : np.ndarray) -> np.ndarray:
    """
    Computes the frequency space of a given dataset.
    :param data: Temporal data containing time and flux
    :return: Dataset containing frequency and power for a given dataset.
    """
    ls = LombScargle(data[0], data[1])
    f, p = ls.autopower(minimum_frequency=0, maximum_frequency=nyquist_f(data), samples_per_peak=100)
    return np.array((f,p))


def get_phases(data : np.ndarray,period : float) -> List[np.ndarray]:
    """
    Computes a list of periods according to a given period.
    :param data: Temporal data containing time and  flux
    :param period: Period where the data is folded
    :return: List of boolean arrays that allow for indexing
    """
    if len(data) != 1:
        x = data[0]
    else:
        x = data

    n_max = int(max(x)/period)

    data_list = []
    for i in range(1,n_max):
        data_list.append(np.logical_and((i-1)*period <= x,i*period > x))

    return data_list


def get_closest_pdm_period(pdm_list : List[float],p_guess : float) -> float:
    """
    Finds the closes pdm period for a given period.
    :param p_guess: Guess for a given timeseries
    :param pdm_list: List of periods found by pdm
    :return: Closest period for the given guess.
    """
    arr = np.array(pdm_list) -p_guess
    arr_sort = np.argsort(np.abs(arr))
    return arr[arr_sort[0]] + p_guess


def get_closest_significant_period(f_data : np.ndarray, p_guess : float) -> float:
    """
    Determines the closest period in frequency space for a given period
    :param f_data: Frequency space of the timeseries
    :param p_guess: Guess of the period
    :return: Closest period in the frequency space.
    """
    for i in [np.inf,-np.inf,np.nan]:
        mask = [f_data[1] != i]

        f_data = np.array((f_data[0][mask],f_data[1][mask]))


    f_data = np.array((f_data[0][1:],f_data[1][1:]))

    peaks,_ = find_peaks(f_data[1],height=np.mean(f_data[1])/2)
    x_data = 1/f_data[0][peaks] - p_guess

    x_sort = np.argsort(np.abs(x_data))
    return x_data[x_sort[0]] + p_guess

def rebin(data : np.ndarray,phase_list : List[np.ndarray]) -> np.ndarray:
    """
    Adds up a rebinned statistic of all phases.
    :param data: Timeseries data
    :param phase_list: List of phase masks
    :return: Array containing rebinned array
    """

    for msk in phase_list:
        phase = np.array((data[0][msk],data[1][msk]))
        phase = normalize_phase(phase)
        bin_y, bin_x, _ = binned_statistic(phase[0], phase[1], statistic='mean', bins=100)
        try:
            val[1] += bin_y
        except UnboundLocalError:
            val = np.array((bin_x,bin_y))

    return val


def boxcar_smoothing(data : np.ndarray,smooth =100) -> np.ndarray:
    """
    Performs boxcar smoothing on a given dataset
    :param data: temporal dataset
    :return: smoothed dataset
    """
    if len(data) != 1:
        y = data[1]
    else:
        y = data

    smoothed_signal = convolve(y, Box1DKernel(smooth))

    if len(data) != 1:
        return np.array((data[0],smoothed_signal))
    else:
        return smoothed_signal

def normalize_phase(data : np.ndarray) -> np.ndarray:
    """
    Normalizes phasedata so that x is between 0 and 1 as well as y
    :param data: temporal dataset
    :return: normalized phase
    """
    x = data[0] - min(data[0])
    x /= max(x)
    y = data[1]
    #y /= max(np.abs(y))
    y -=np.mean(y)

    return np.array((x,y))


def perform_classification(data : np.ndarray) -> bool:
    """
    Performs the classification for a given dataset.
    :param data: Timeseries data. Should be already reduced.
    :return: Boolean value determining the data.
    """
    pass
