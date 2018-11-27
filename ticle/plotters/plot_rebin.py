import matplotlib.pyplot as pl
import os
import numpy as np

from ticle.data.dataHandler import normalizeData,load_file
from ticle.analysis.analysis import get_significant_periods,boxcar_smoothing,count_zero_crossings\
    ,get_closest_pdm_period,get_lomb_scargle,get_closest_significant_period,get_phases,rebin

pl.rc('xtick', labelsize='x-small')
pl.rc('ytick', labelsize='x-small')
pl.rc('font', family='serif')
pl.rcParams.update({'font.size': 20})
pl.tight_layout()

path = os.getcwd()
rebin_result_dir = f"{path}/results/rebin_result"

try:
    os.makedirs(rebin_result_dir)
except FileExistsError:
    pass

data_dir = f"{path}/data/"
data_list_file = f"{data_dir}/dataList.txt"
data_list = np.loadtxt(data_list_file)

for data in data_list:
    star = f"0{int(data[0])}"
    file_name = f"{data_dir}/{star}/{star}_LC_destepped.txt"
    res_dir = f"{rebin_result_dir}/{star}"

    try:
        os.mkdir(res_dir)
    except FileExistsError:
        pass

    t_series = load_file(file_name)
    t_series = normalizeData(t_series)

    smoothed = boxcar_smoothing(t_series)

    zc = count_zero_crossings(smoothed)
    p_guess = float(2 * max(smoothed[0]) / zc)

    pdm = get_significant_periods(t_series, 20)

    p_pdm = get_closest_pdm_period(pdm,p_guess)

    f_data = get_lomb_scargle(t_series)
    p = f_data[1][f_data[0] < (1 / p_pdm) * 1.5]
    f = f_data[0][f_data[0] < (1 / p_pdm) * 1.5]

    p_f = get_closest_significant_period(np.array((f,p)),p_pdm)

    phaseList = get_phases(t_series,p_f)

    rebin_data = rebin(t_series,phaseList)
    rebin_data = np.array((rebin_data[0][1:],rebin_data[1]))
    boxcar_rebin = boxcar_smoothing(rebin_data,20)

    fig_result_rebin = pl.figure(figsize=(10, 7))
    pl.plot(rebin_data[0],rebin_data[1],'x',markersize=3,color='k',label='Rebinned data')
    pl.plot(boxcar_rebin[0],boxcar_rebin[1],color='k',label='Smoothed rebin',alpha=0.5)
    pl.legend()
    pl.xlabel("Phase")
    pl.ylabel("Flux")
    pl.title(f"Rebinned data {star} - result")
    fig_result_rebin.savefig(f"{res_dir}/{star}_rebin_result.pdf")

    phaseList = get_phases(t_series,data[2])

    rebin_data = rebin(t_series,phaseList)
    rebin_data = np.array((rebin_data[0][1:],rebin_data[1]))
    boxcar_rebin = boxcar_smoothing(rebin_data,20)

    fig_result_rebin = pl.figure(figsize=(10, 7))
    pl.plot(rebin_data[0],rebin_data[1],'x',markersize=3,color='k',label='Rebinned data')
    pl.plot(boxcar_rebin[0],boxcar_rebin[1],color='k',label='Smoothed rebin',alpha=0.5)
    pl.legend()
    pl.xlabel("Phase")
    pl.ylabel("Flux")
    pl.title(f"Rebinned data {star} - Literature")
    fig_result_rebin.savefig(f"{res_dir}/{star}_rebin_literature.pdf")
