import matplotlib.pyplot as pl
import os
import numpy as np

from ticle.data.dataHandler import normalizeData,load_file
from ticle.analysis.analysis import get_phases,normalize_phase

pl.rc('xtick', labelsize='x-small')
pl.rc('ytick', labelsize='x-small')
pl.rc('font', family='serif')
pl.rcParams.update({'font.size': 20})
pl.tight_layout()

path = os.getcwd()
phase_dir = f"{path}/results/phase_plots"

try:
    os.makedirs(phase_dir)
except FileExistsError:
    pass

data_dir = f"{path}/data/"
data_list_file = f"{data_dir}/dataList.txt"
data_list = np.loadtxt(data_list_file)

for data in data_list:
    star = f"0{int(data[0])}"
    file_name = f"{data_dir}/{star}/{star}_LC_destepped.txt"
    res_dir = f"{phase_dir}/{star}"

    try:
        os.mkdir(res_dir)
    except FileExistsError:
        pass

    t_series = load_file(file_name)
    t_series = normalizeData(t_series)

    p = [(f"Phaseplot {star} - literature","literature",data[2]),
         (f"Phaseplot {star} - P={data[1]} days",f"result",data[1])]


    for title,save_text,period in p:
        masks = get_phases(t_series,period)

        fig_phase = pl.figure(figsize=(10,7))

        for i in masks:
            plot_data = normalize_phase(np.array((t_series[0][i],t_series[1][i])))
            pl.plot(plot_data[0],plot_data[1],linewidth = 1)

        pl.xlabel("Phase")
        pl.ylabel("Flux")
        pl.title(title)
        fig_phase.savefig(f"{res_dir}/{star}_{save_text}_phase_.pdf")

        fig_lightcurve = pl.figure(figsize=(10,7))

        for i in masks:
            pl.plot(t_series[0][i],t_series[1][i],linewidth = 1)

        pl.xlabel("Period(days)")
        pl.ylabel("Flux")
        pl.title(f"{star} Lightcurve {save_text}")
        fig_lightcurve.savefig(f"{res_dir}/{star}_{save_text}_lightcurve.pdf")