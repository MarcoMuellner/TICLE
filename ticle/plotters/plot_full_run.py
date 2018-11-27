import matplotlib.pyplot as pl
import os
import numpy as np

from ticle.data.dataHandler import normalizeData,load_file
from ticle.analysis.analysis import get_significant_periods,boxcar_smoothing,count_zero_crossings\
    ,get_closest_pdm_period,get_lomb_scargle,get_closest_significant_period,get_phases,normalize_phase

pl.rc('xtick', labelsize='x-small')
pl.rc('ytick', labelsize='x-small')
pl.rc('font', family='serif')
#pl.rcParams.update({'font.size': 20})
pl.tight_layout()

path = os.getcwd()
full_run_dir = f"{path}/results/full_run"

try:
    os.makedirs(full_run_dir)
except FileExistsError:
    pass

data_dir = f"{path}/data/"
data_list_file = f"{data_dir}/dataList.txt"
data_list = np.loadtxt(data_list_file)

for data in data_list:
    star = f"0{int(data[0])}"
    file_name = f"{data_dir}/{star}/{star}_LC_destepped.txt"
    res_dir = f"{full_run_dir}/{star}"

    try:
        os.mkdir(res_dir)
    except FileExistsError:
        pass

    t_series = load_file(file_name)
    t_series = normalizeData(t_series)

    smoothed = boxcar_smoothing(t_series)
    zc, ind = count_zero_crossings(smoothed, True)
    p_guess = float(2 * max(smoothed[0]) / zc)

    fig,axes = pl.subplots(4,2,figsize=(13,18))

    axes[0,0].plot(smoothed[0], smoothed[1], color='k')
    axes[0,0].set_xlabel("Time(days)")
    axes[0,0].set_ylabel("Flux")
    axes[0,0].axhline(y=0, linestyle='dashed', color='k', alpha=0.6)
    axes[0,0].set_title(f"Boxcar smoothed {star}")

    axes[1,0].plot(smoothed[0], smoothed[1], color='k')
    axes[1,0].axhline(y=0, linestyle='dashed', color='k', alpha=0.6)
    for i in ind[0]:
        axes[1,0].axvline(x=smoothed[0][i], linestyle='dotted', color='red')

    axes[1,0].set_ylabel("Flux")
    axes[1,0].set_title(f"Zero crossings {star}")
    axes[1,0].text(0.5, -0.1, rf"$P_{{guess}}={p_guess}$ days",
            size=14, ha='center', transform=axes[1,0].transAxes)

    pdm = get_significant_periods(t_series, 20, True)
    p_pdm = get_closest_pdm_period(pdm['nbestperiods'], p_guess)
    p_pdm_string = "%.2f" % float(p_pdm)

    periods = pdm['periods']
    sigma_vals = pdm['lspvals'][periods * 2 < max(t_series[0])]
    periods = periods[periods * 2 < max(t_series[0])]

    axes[2,0].plot(t_series[0], t_series[1], color='k')
    axes[2,0].set_xlabel("Time(days)")
    axes[2,0].set_ylabel("Flux")
    axes[2,0].axhline(y=0, linestyle='dashed', color='k', alpha=0.6)
    axes[2,0].set_title(f"{star}")

    axes[3,0].plot(periods, sigma_vals, color='k', markersize=3, alpha=0.5)
    axes[3,0].plot(periods, sigma_vals, 'x', color='k', markersize=3)
    p_guess_str = "%.2f" % float(p_guess)
    axes[3,0].axvline(x=p_guess, color='blue', alpha=0.6, linestyle='dashed', label=rf"$P_{{guess}}={p_guess_str}$ days")
    axes[3,0].axvline(x=p_pdm, color='cyan', alpha=0.6, linestyle='dashed', label=rf"$P_{{pdm}}={p_pdm_string}$ days")
    axes[3,0].set_xlabel(r"Period(days)")
    axes[3,0].set_ylabel(r"$\Theta_{pdm}$")
    axes[3,0].legend()
    axes[3,0].set_title(f"PDM {star}")

    f_data = get_lomb_scargle(t_series)
    p = f_data[1][f_data[0] < (1 / p_pdm) * 1.5]
    f = f_data[0][f_data[0] < (1 / p_pdm) * 1.5]

    p_f = get_closest_significant_period(np.array((f,p)),p_pdm)
    p_f_string = "%.2f" % float(p_f)

    axes[1,1].plot(f,p,color='k')
    axes[1,1].set_xlabel("Frequency (c/d)")
    axes[1,1].set_ylabel("Power")
    axes[1,1].axvline(x=1 / p_pdm, label=f"1/$P_{{pdm}}$=1/{p_pdm_string}",color='blue',linestyle='dashed',alpha=0.7)
    axes[1,1].axvline(x=1 / p_f, label=f"1/$P_{{f}}$=1/{p_f_string}", color='cyan', linestyle='dashed')
    axes[1,1].legend()

    masks = get_phases(t_series, p_f)

    for i in masks:
        plot_data = normalize_phase(np.array((t_series[0][i], t_series[1][i])))
        axes[2,1].plot(plot_data[0], plot_data[1], linewidth=1)

    axes[2,1].set_xlabel("Phase")
    axes[2,1].set_ylabel("Flux")
    axes[2,1].set_title(f"Phaseplot {star} - P={p_f_string} days")

    for i in masks:
        axes[3,1].plot(t_series[0][i], t_series[1][i], linewidth=1)

    axes[3,1].set_xlabel("Period(days)")
    axes[3,1].set_ylabel("Flux")
    axes[3,1].set_title(f"{star} Lightcurve")
    pl.show()