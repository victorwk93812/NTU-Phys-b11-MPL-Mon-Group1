# import math
# import copy
# import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import csv
# import uncertainties as uct
# from uncertainties import unumpy as unp
import scipy
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
# from scipy.signal import savgol_filter, medfilt

matplotlib.use("Qt5Agg")  # Force the Qt backend

fnames = [f"SE_{i}.csv" for i in range(1, 4)]
NMR_time = []
NMR_amp = []

for i in range(3):
    df = pd.read_csv(f"../data/{fnames[i]}")
    if "CH2(V)" in df:
        NMR_amp.append(df["CH2(V)"])
        NMR_time.append(np.linspace(df["Time(s)"].iloc[0], df["Time(s)"].iloc[-1], len(NMR_amp[-1])))

def moving_average(x, w):
    return np.convolve(x, np.ones(w)/w, mode='same')

tind_l, tind_r = 130000, 450000
NMR_time_sliced = [time[tind_l:tind_r] for time in NMR_time]
NMR_amp_sliced = [amp[tind_l:tind_r] for amp in NMR_amp]
# plt.plot(NMR_time[0], NMR_amp[0])

wsize = [10000, 20000, 35000] # average of "wsize" time indexes
NMR_amp_smth = [moving_average(NMR_amp_sliced[i], wsize[i]) for i in range(3)]


NMR_sig_peaks = []
for i in range(3):
    peaks, _ = find_peaks(NMR_amp_smth[i], prominence = 0.1)
    NMR_sig_peaks.append(peaks)

figSE, axsSE = plt.subplots(3, figsize = (10, 5))
T2fit_time = []
T2fit_amp = []
for i, axs in enumerate(axsSE):
    axs.plot(NMR_time_sliced[i], NMR_amp_smth[i])
    axs.scatter(NMR_time_sliced[i][NMR_sig_peaks[i]], NMR_amp_smth[i][NMR_sig_peaks[i]], color = "orange")
    axs.annotate(f"{NMR_time_sliced[i][NMR_sig_peaks[i][1]]:.2f}", (NMR_time_sliced[i][NMR_sig_peaks[i][1]], NMR_amp_smth[i][NMR_sig_peaks[i][1]]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8)
    T2fit_time.append(NMR_time_sliced[i][NMR_sig_peaks[i][1]])
    T2fit_amp.append(NMR_amp_smth[i][NMR_sig_peaks[i][1]])
plt.show()

def magmom_T2_eq(time, Mz, T2):
    return Mz * np.exp(-time/T2)

T2popt, T2pcov = curve_fit(magmom_T2_eq, T2fit_time, T2fit_amp)
print(T2popt)




