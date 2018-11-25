import matplotlib.pyplot as pl
import os
import numpy as np

from ticle.data.dataHandler import normalizeData,load_file
from ticle.analysis.analysis import get_significant_periods

pl.rc('xtick', labelsize='x-small')
pl.rc('ytick', labelsize='x-small')
pl.rc('font', family='serif')

path = os.getcwd()
sigma_pdm_dir = f"{path}/results/sigma_pdm"

try:
    os.makedirs(sigma_pdm_dir)
except FileExistsError:
    pass

data_dir = f"{path}/data/"
data_list_file = f"{data_dir}/dataList.txt"
data_list = np.loadtxt(data_list_file)

for data in data_list:
    star = f"0{int(data[0])}"
    file_name = f"{data_dir}/{star}/{star}_LC_destepped.txt"
    res_dir = f"{sigma_pdm_dir}/{star}"

    try:
        os.mkdir(res_dir)
    except FileExistsError:
        pass

    t_series = load_file(file_name)
    t_series = normalizeData(t_series)

    pdm = get_significant_periods(t_series, 20, True)

    periods = pdm['periods']
    sigma_vals = pdm['lspvals'][periods * 2 < max(t_series[0])]
    periods = periods[periods * 2 < max(t_series[0])]

    lightcurve_fig = pl.figure(figsize=(10,7))
    pl.plot(t_series[0],t_series[1],color='k')
    pl.xlabel("Time(days)")
    pl.ylabel("Flux")
    pl.axhline(y=0,linestyle='dashed',color='k',alpha=0.6)
    pl.title(f"{star}")

    pdm_fig = pl.figure(figsize=(10, 7))
    pl.plot(periods, sigma_vals, color='k', markersize=3,alpha=0.5)
    pl.plot(periods,sigma_vals,'x',color='k',markersize=3)
    p_guess = "%.2f" % float(data[1])
    pl.axvline(x=data[1],color='blue',alpha=0.6,linestyle='dashed',label=rf"$P_{{guess}}={p_guess}$ days")
    pl.xlabel("Period(days)")
    pl.ylabel(r"$\Theta_{pdm}$")
    pl.legend()
    pl.title(f"PDM {star}")

    lightcurve_fig.savefig(f"{res_dir}/{star}_lightcurve.pdf")
    pdm_fig.savefig(f"{res_dir}/{star}_pdm.pdf")


