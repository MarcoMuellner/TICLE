import pytest
import numpy as np

from ticle.analysis.analysis import get_significant_periods,count_zero_crossings,get_periodogram,get_phases\
    ,get_closest_pdm_period,get_closest_significant_period,find_most_common_diff

x = np.linspace(0,100,num=1000)

@pytest.mark.parametrize("testPeriods",[[1],[5],[5,2,10],[5,10,15,20]])
def test_get_significant_periods(testPeriods):
    y = np.zeros(x.shape)

    for p in testPeriods:
        y +=np.sin(p*x)

    n = len(testPeriods)

    periods = get_significant_periods(np.array((x,y)),n)

    for i in periods:
        assert len(np.where(np.abs(np.array(testPeriods) - i*np.pi)<10**-2))

@pytest.mark.parametrize("testFunctions",[np.sin(x*np.pi),np.sin(x*np.pi)+5,np.sin(x*np.pi)-5])
def test_count_zero_crossings(testFunctions):
    zc = count_zero_crossings(np.array((x,testFunctions)))
    val = int(max(x))
    assert  zc == val


@pytest.mark.parametrize("testPeriods",[[5],[5,2,10],[5,10,15,20]])
def test_periodogram(testPeriods):
    y = np.zeros(x.shape)

    for p in testPeriods:
        y += np.sin(p * x)

    f_data = get_periodogram(np.array((x,y)))

    assert np.all(f_data[0] > 0)

    for i in testPeriods:
        arg = np.argsort(f_data[1])
        maxF = 1/f_data[1][arg][0:len(testPeriods)]
        assert len(np.where(np.abs(maxF -i) < 10*-2)) == 1


@pytest.mark.parametrize("testFunctions", [np.sin(x * np.pi), np.sin(x * np.pi) + 5, np.sin(x * np.pi) - 5])
def test_get_phases(testFunctions):
    phaseList = get_phases(np.array((x,testFunctions)),2)

    assert len(phaseList) == 49

    equalFlag = True

    for i in range(1,len(phaseList)):
        assert len(testFunctions[phaseList[i]]) == len(testFunctions[phaseList[i]])
        equalFlag = equalFlag and np.all(testFunctions[phaseList[i]] - testFunctions[phaseList[i-1]] < 10**-2)
    assert equalFlag


@pytest.mark.parametrize("testFunctions", [np.sin(x * np.pi), np.sin(x * np.pi) + 5, np.sin(x * np.pi) - 5])
def test_get_closest_pdm_period(testFunctions):
    data = np.array((x,testFunctions))
    zc = count_zero_crossings(data)

    p_guess = 2*max(x)/zc

    pdm = get_significant_periods(data,10)

    p = get_closest_pdm_period(pdm,p_guess)

    assert abs(p -2) < 10**-3


@pytest.mark.parametrize("testFunctions", [np.sin(x * np.pi), np.sin(x * np.pi) + 5, np.sin(x * np.pi) - 5])
def test_get_closes_significant_period(testFunctions):
    data = np.array((x, testFunctions))

    zc = count_zero_crossings(data)
    p_guess = 2 * max(x) / zc

    fData = get_periodogram(data)

    pdm = get_significant_periods(data, 10)

    p = get_closest_pdm_period(pdm, p_guess)

    p = get_closest_significant_period(fData,p)

    assert abs(p-2) < 10**-2


def test_find_most_common_diff():
    assert abs(x[1] - x[0] - find_most_common_diff(x)) < 10 ** -5

