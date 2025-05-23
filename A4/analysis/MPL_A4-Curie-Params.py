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

fnames = ["865_11_01_13s_10A0.csv", "865_11_01_13s_13A0.csv", "865_11_01_13s_15A0.csv", "865_11_01_13s_18A0.csv", "865_11_01_13s_20A0.csv", "865_11_01_13s_23A0.csv", "865_11_01_13s_25A0.csv", "865_11_01_13s_28A0.csv"] 
pcur = [1, 1.3, 1.5, 1.8, 2, 2.3, 2.5, 2.8]

NMR_amp = []
NMR_time = []

for fname in fnames:
    df = pd.read_csv(f"../data/{fname}")
    if "CH2(V)" in df:
        NMR_amp.append(df["CH2(V)"])
        NMR_time.append(np.linspace(df["Time(s)"].iloc[0], df["Time(s)"].iloc[-1], len(NMR_amp[-1])))

tind_l, tind_r = 200000, 800000
NMR_time_sliced = [time[tind_l:tind_r] for time in NMR_time]
NMR_amp_sliced = [amp[tind_l:tind_r] for amp in NMR_amp]

# Debug NMR amplitude signal
# plt.plot(NMR_time[0], NMR_amp[0])
# plt.plot(NMR_time_sliced[0], NMR_amp_sliced[0])
# plt.show()

def magmom_T2star_eq(time, Mxy, T2star):
    return Mxy * np.exp(-time/T2star)

figT2star, axsT2star = plt.subplots(4, 2, figsize = (12, 10))
axsT2star = axsT2star.flatten()

T2star_time_span = NMR_time_sliced[0]

print("---Curie M0, T2* Fit---")
for i, axs in enumerate(axsT2star):
    T2starpopt, T2starpcov = curve_fit(magmom_T2star_eq, T2star_time_span, NMR_amp_sliced[i])
    T2starperr = np.sqrt(np.diag(T2starpcov))
    print(f"Curie pol. current {pcur[i]} A fit M0: {T2starpopt[0]:.3f} ± {T2starperr[0]:.3f} s")
    print(f"Curie pol. current {pcur[i]} A fit T2*: {T2starpopt[1]:.3f} ± {T2starperr[1]:.3f} s")
    T2star_fit_curve = magmom_T2star_eq(T2star_time_span, *T2starpopt)

    axs.plot(T2star_time_span, NMR_amp_sliced[i], color = "blue", label = f"Polarize current {pcur[i]} A NMR amplitude data")
    axs.plot(T2star_time_span, T2star_fit_curve, color = "red", label = "Fit curve")
    axs.legend()
    axs.grid()
    axs.set_title(f"Polarization Current {pcur[i]} A")
figT2star.suptitle("Curie Law Exp. NMR Amplitude Signal T2star Decay Fit")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, 
                wspace=0.1, hspace=0.5)
figT2star.savefig("../pics/Curie-T2star-Decay-Fit.png")
plt.show()

