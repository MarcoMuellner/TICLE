import matplotlib.pyplot as pl
import os
import numpy as np

from ticle.data.dataHandler import normalizeData,load_file
from ticle.analysis.analysis import get_significant_periods,boxcar_smoothing,count_zero_crossings\
    ,get_closest_pdm_period,get_lomb_scargle,get_closest_significant_period

pl.rc('xtick', labelsize='x-small')
pl.rc('ytick', labelsize='x-small')
pl.rc('font', family='serif')
pl.rcParams.update({'font.size': 20})
pl.tight_layout()

path = os.getcwd()
f_space_dir = f"{path}/results/f_space"

try:
    os.makedirs(f_space_dir)
except FileExistsError:
    pass

data_dir = f"{path}/data/"
data_list_file = f"{data_dir}/dataList.txt"
data_list = np.loadtxt(data_list_file)

for data in data_list:
    star = f"0{int(data[0])}"
    file_name = f"{data_dir}/{star}/{star}_LC_destepped.txt"
    res_dir = f"{f_space_dir}/{star}"

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
    p_pdm_string = "%.2f" % float(p_pdm)

    f_data = get_lomb_scargle(t_series)
    p = f_data[1][f_data[0] < (1 / p_pdm) * 1.5]
    f = f_data[0][f_data[0] < (1 / p_pdm) * 1.5]

    p_f = get_closest_significant_period(np.array((f,p)),p_pdm)
    p_f_string = "%.2f" % float(p_f)

    f_space_fig = pl.figure(figsize=(10, 7))

    pl.plot(f,p,color='k')
    pl.xlabel(r"Frequency ($d^{-1}$)")
    pl.ylabel("Power")
    pl.axvline(x=1 / p_pdm, label=f"1/$P_{{pdm}}$=1/{p_pdm_string}",color='blue',linestyle='dashed',alpha=0.7)
    pl.axvline(x=1 / p_f, label=f"1/$P_{{f}}$=1/{p_f_string}", color='cyan', linestyle='dashed')
    pl.legend()
    f_space_fig.savefig(f"{res_dir}/{star}_frequency_space.pdf")
