import matplotlib.pyplot as pl
import os
import numpy as np

from ticle.data.dataHandler import normalizeData,load_file
from ticle.analysis.analysis import count_zero_crossings,boxcar_smoothing

pl.rc('xtick', labelsize='x-small')
pl.rc('ytick', labelsize='x-small')
pl.rc('font', family='serif')
pl.rcParams.update({'font.size': 20})
pl.tight_layout()

path = os.getcwd()
zc_dir = f"{path}/results/zc"

try:
    os.makedirs(zc_dir)
except FileExistsError:
    pass

data_dir = f"{path}/data/"
data_list_file = f"{data_dir}/dataList.txt"
data_list = np.loadtxt(data_list_file)

for data in data_list:
    star = f"0{int(data[0])}"
    file_name = f"{data_dir}/{star}/{star}_LC_destepped.txt"
    res_dir = f"{zc_dir}/{star}"

    try:
        os.mkdir(res_dir)
    except FileExistsError:
        pass

    t_series = load_file(file_name)
    t_series = normalizeData(t_series)

    smoothed = boxcar_smoothing(t_series)

    zc,ind = count_zero_crossings(smoothed,True)
    p_guess = "%.2f" % float(2*max(smoothed[0])/zc)
    print(zc)
    print(p_guess)

    smoothed_fig = pl.figure(figsize=(10, 7))
    pl.plot(smoothed[0],smoothed[1],color='k')
    pl.xlabel("Time(days)")
    pl.ylabel("Flux")
    pl.axhline(y=0,linestyle='dashed',color='k',alpha=0.6)
    pl.title(f"Boxcar smoothed {star}")

    text =  [("description",rf"$P_{{guess}}$=$2T/n_{{zc}}$"),
             ("zc_result",rf"$P_{{guess}}={p_guess}$ days$\leftrightarrow P_{{literature}} = {data[2]}$ days")]

    for descr,txt in text:
        fig = pl.figure(figsize=(10, 7))
        pl.plot(smoothed[0], smoothed[1], color='k')
        pl.axhline(y=0, linestyle='dashed', color='k', alpha=0.6)
        for i in ind[0]:
            pl.axvline(x=smoothed[0][i],linestyle='dotted',color='red')

        pl.ylabel("Flux")
        pl.title(f"Zero crossings {star}")
        pl.text(0.5, -0.1, txt,
                size=14, ha='center', transform=pl.gca().transAxes)
        fig.savefig(f"{res_dir}/{star}_{descr}.pdf")

    smoothed_fig.savefig(f"{res_dir}/{star}_smoothed_lightcurve.pdf")


