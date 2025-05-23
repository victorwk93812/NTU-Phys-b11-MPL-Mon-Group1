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

## Fit T1
## Use FFT (MATH1) peak at plztn. freq. to fit the mag. mom. T.E. eq.
fnames = ["865_11_02-05s0.csv", "865_11_02-1s0.csv", "865_11_02-2s0.csv", "865_11_01.csv", "865_11_02-4s0.csv", "865_11_02-5s0.csv", "865_11_02-13s0.csv"] 
ptime = [0.5, 1, 2, 3, 4, 5, 13]
FFT_used_ptime = [1, 2, 3, 4, 5, 13]
T2star_used_ptime = [0.5, 1, 2, 4, 5, 13]
# FFT_sig = []
# FFT_freq = []
NMR_amp = []
NMR_sig = []
NMR_time = []
for fname in fnames:
    df = pd.read_csv(f"../data/{fname}")
    # Obtain FFT sinal from oscilloscope
    # Abandoned since we have no info about the frequency range of the spectrum
    # if "MATH1(Vrms)" in df:
    #     FFT_sig.append(df["MATH1(Vrms)"])
    #     FFT_freq.append(np.linspace(0, 200000, len(FFT_sig[-1])))
    if "CH1(V)" in df:
        NMR_sig.append(df["CH1(V)"])
    if "CH2(V)" in df:
        NMR_amp.append(df["CH2(V)"])
        NMR_time.append(np.linspace(df["Time(s)"].iloc[0], df["Time(s)"].iloc[-1], len(NMR_amp[-1])))


tind_l, tind_r = 130000, 450000
NMR_time_sliced = [time[tind_l:tind_r] for time in NMR_time]
NMR_sig_sliced = [sig[tind_l:tind_r] for sig in NMR_sig]
NMR_amp_sliced = [amp[tind_l:tind_r] for amp in NMR_amp]

# Debug NMR signal figure
# plt.plot(NMR_time[0], NMR_amp[0])
# plt.plot(NMR_time_sliced[3], NMR_sig_sliced[3])
# plt.show()

N = tind_r - tind_l # number of signal data points
dt = NMR_time_sliced[0][1] - NMR_time_sliced[0][0] # FFT sampling rate
FFT_sig = [fft(signal)/N for signal in NMR_sig_sliced]
FFT_freq = [fftfreq(N, dt) for i in range(len(NMR_sig_sliced))]

# Keep only positive half of the spectrum
FFT_freq = [freq[:N//2] for freq in FFT_freq]
FFT_sig = [np.abs(signal[:N//2]) for signal in FFT_sig]

# Debug FFT signal figure
# plt.plot(FFT_freq[3], FFT_sig[3])
# plt.show()

# Mask with frequency region
freq_l, freq_r = 2000, 2500
mask = (FFT_freq[0] >= freq_l) & (FFT_freq[0] <= freq_r)
FFT_freq = [freq[mask] for freq in FFT_freq]
FFT_sig = [sig[mask] for sig in FFT_sig]
FFT_peaks = []
FFT_prec_idxs = [None] * 7

for signal in FFT_sig:
    peaks, _ = find_peaks(signal, prominence = 0.06)
    FFT_peaks.append(peaks)

figFFT, axsFFT = plt.subplots(2, 3, figsize = (10, 5))
i = 1 
axsFFT = axsFFT.flatten()
for ax in axsFFT:
    ax.plot(FFT_freq[i], FFT_sig[i])

    # ax.plot(FFT_freq[i][FFT_peaks[i]], FFT_sig[i][FFT_peaks[i]], "^g", label = "Peaks")

    # Find the peak near 2261.5 and mark it as precession signal
    FFT_peak_freqs = FFT_freq[i][FFT_peaks[i]]
    peak_freq_est = 2261.5 # estimated peak frequency
    idx = (np.abs(FFT_peak_freqs - peak_freq_est)).argmin()
    FFT_prec_idxs[i] = FFT_peaks[i][idx]
    ax.plot(FFT_freq[i][FFT_peaks[i][idx]], FFT_sig[i][FFT_peaks[i][idx]], "or", label = "Precession Signal")
    ax.annotate(f"{FFT_freq[i][FFT_peaks[i][idx]]:.2f} Hz", (FFT_freq[i][FFT_peaks[i][idx]], FFT_sig[i][FFT_peaks[i][idx]]), textcoords="offset points", xytext=(0,-10), ha='center', fontsize=8)
    ax.grid()
    ax.legend()
    ax.set_title(f"Polarize {FFT_used_ptime[i - 1]} s")
    i = i + 1
figFFT.suptitle("Precession Signal Detection in NMR Amplitude Signals")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, 
                wspace=0.1, hspace=0.5)
figFFT.savefig("../pics/Peak-Detection.png")
plt.show()

T1_fit_time = ptime[1:]
T1_magmom = []
print("---NMR Amplitude Signal FFT Precession Frequencies---")
for i in range(1, 7):
    T1_magmom.append(FFT_sig[i][FFT_prec_idxs[i]])
    print(f"Polarization time: {ptime[i]}, Precession Frequency: {FFT_freq[i][FFT_prec_idxs[i]]:.2f}, FFT Magnitude: {FFT_sig[i][FFT_prec_idxs[i]]:.2f}")

def magmom_T1_eq(time, Mz, T1):
    return Mz * (1 - np.exp(-time/T1))

T1popt, T1pcov = curve_fit(magmom_T1_eq, T1_fit_time, T1_magmom)
T1perr = np.sqrt(np.diag(T1pcov))
print("---T1 Fit---")
print(f"Fit T1: {T1popt[1]:.3f} ± {T1perr[1]:.3f} s")

T1_time_span = np.linspace(0, 14, 1000)
T1_fit_curve = magmom_T1_eq(T1_time_span, *T1popt)

fig_T1fit, axs_T1fit = plt.subplots(1, figsize = (10, 5))
axs_T1fit.plot(T1_time_span, T1_fit_curve, label = "Fit curve")
axs_T1fit.scatter(T1_fit_time, T1_magmom, color = "red", label = "Amplitude of precession frequency")
axs_T1fit.set_title("Amplitude of Precession Frequency to Polarization Time")
axs_T1fit.legend()
fig_T1fit.savefig("../pics/T1-fit.png")
plt.show()

## Fit T2
## Use find_peaks on CH2 and then fit the T2 eq.

tind_l, tind_r = 160000, 350000
NMR_time_sliced = [time[tind_l:tind_r] for time in NMR_time]
NMR_sig_sliced = [sig[tind_l:tind_r] for sig in NMR_sig]
NMR_amp_sliced = [amp[tind_l:tind_r] for amp in NMR_amp]

# Debug NMR amplitude signal
# plt.plot(NMR_time_sliced[3], NMR_amp_sliced[3])
# plt.show()

def magmom_T2star_eq(time, Mxy, T2star):
    return Mxy * np.exp(-time/T2star)

figT2star, axsT2star = plt.subplots(3, figsize = (12, 8))
axsT2star = axsT2star.flatten()

T2star_time_span = NMR_time_sliced[0]

print("---T2* Fit---")
for i, axs in enumerate(axsT2star):
    T2starpopt, T2starpcov = curve_fit(magmom_T2star_eq, T2star_time_span, NMR_amp_sliced[i + 3])
    T2starperr = np.sqrt(np.diag(T2starpcov))
    print(f"Fit T2star: {T2starpopt[1]:.3f} ± {T2starperr[1]:.3f} s")
    T2star_fit_curve = magmom_T2star_eq(T2star_time_span, *T2starpopt)

    axs.plot(T2star_time_span, NMR_amp_sliced[i + 3], color = "blue", label = f"Polarize {T2star_used_ptime[i + 3]} s NMR amplitude data")
    axs.plot(T2star_time_span, T2star_fit_curve, color = "red", label = "Fit curve")
    axs.set_title(f"Polarize {T2star_used_ptime[i + 3]} s Fit")
    axs.legend()
    axs.grid()
figT2star.suptitle("NMR Amplitude Signal T2star Decay Fit")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, 
                wspace=0.1, hspace=0.5)
figT2star.savefig("../pics/T2star-fit.png")
plt.show()

