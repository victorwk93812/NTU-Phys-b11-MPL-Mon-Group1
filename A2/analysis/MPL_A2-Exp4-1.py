# import copy
# import numpy as np
# import pandas as pd
# import csv
# import uncertainties as uct
# from uncertainties import unumpy as unp
# import scipy
# from scipy.optimize import curve_fit
# from scipy.signal import savgol_filter, medfilt
import math
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

matplotlib.use("Qt5Agg")  # Force the Qt backend

class InvalidLineFormatError(Exception):
    """Exception raised when a line does not contain exactly two floats."""
    
    def __init__(self, line, tokens):
        self.line = line
        self.tokens = tokens
        super().__init__(f"Invalid line format: {line!r} (parsed: {tokens})")

@dataclass
class Spectrum:
    # target data file name relative to the data directory "../data/"
    fname: str
    # tube length (mm)
    tubelen: float
    # peak windowing size
    wsize: int = 5
    # peak threshold amplitude
    ampthrs: int | None = None
    # data size
    dsize: int = field(init = False)
    # base frequency (n = 1) = c/2L (Hz)
    basefreq: float = field(init = False)
    # base wave number (n = 1) = pi/L (1/mm)
    basewavenum: float = field(init = False)
    # plots
    plots: list[None] = field(init = False, default_factory = list)
    # frequency data points
    freq: list[float] = field(init = False, default_factory = list)
    # amplitude data points
    amp: list[float] = field(init = False, default_factory = list)
    # list of peaks in format (peak_freq, peak_amp)
    pklist: list[tuple[float, float]] = field(init = False, default_factory = list)

    # Following are shared among all Spectrum instantiates
    # data directory
    datadir = "../data/"
    # figures directory
    picsdir = "../pics/"
    # speed of sound = 34300 mm/s
    c = 34300

    def parse_line(self, line: str):
        tokens = line.split()  # Split by spaces/tabs

        try:
            floats = [float(x) for x in tokens]  # Try converting to float
        except ValueError:
            raise InvalidLineFormatError(line, tokens)  # Non-float detected

        if len(floats) != 2:  # Ensure exactly two floats
            raise InvalidLineFormatError(line, floats)

        return tuple(floats)  # Return as a tuple (float1, float2)

    def find_peaks(self):
        """Find resonance frequencies of data."""
        whalf = self.wsize // 2
        for i in range(0, self.dsize - self.wsize):
            if self.amp[i + whalf] == max(self.amp[i:i + self.wsize]):
                if type(self.ampthrs) == int and self.amp[i + whalf] >= self.ampthrs:
                    self.pklist.append((self.freq[i + whalf], self.amp[i + whalf]))

    def __post_init__(self):
        with open(Spectrum.datadir + self.fname, 'r', encoding = "utf-8") as file:
            while True:
                line = file.readline()
                if not line:
                    break

                try:
                    pfreq, pamp = self.parse_line(line)
                except InvalidLineFormatError as e:
                    print(f"Invalid Data Format {e}")
                    exit(1)
                self.freq.append(pfreq)
                self.amp.append(pamp)

        self.basefreq = Spectrum.c / (2 * self.tubelen)
        self.basewavenum = math.pi / self.tubelen
        self.dsize = len(self.freq)
        self.find_peaks()

    def dbg(self):
        """Print first 5 sets of data."""
        print(self.freq[0:5], self.amp[0:5])

    def peaks(self) -> list[tuple[float, float]]:
        """Return frequency peaks of this spectrum."""
        return self.pklist

    def plot_spectrum(self, figname: str, tname: str, annotate: bool = False, showfig: bool = True):
        """Plot the amplitude vs frequency spectrum."""
        # Create the plot
        plt.figure(figsize=(10, 5))  # Set figure size
        plt.plot(self.freq, self.amp, label="Amplitude vs Frequency")  # Plot data

        x_bot, x_top = plt.xlim()
        x_len = x_top - x_bot
        y_bot, y_top = plt.ylim()
        y_len = y_top - y_bot

        # Annotate peaks
        if annotate:
            for peak in self.pklist:
                plt.annotate(f"{peak[0]:.1f} Hz", 
                             peak, 
                             textcoords="offset points",
                             xytext=(0, 10), 
                             ha='center', 
                             fontsize=9, 
                             color='red')

        # Mark peaks with vertical lines
        modenum = 1
        resfreq = self.basefreq
        while(modenum * self.basefreq <= self.freq[-1]):
            plt.axvline(x = resfreq, color='orange', linestyle='--', alpha=0.7)
            # Define a proxy line (doesn't appear in plot, only legend)
            peak_proxy = mlines.Line2D([], [], color='orange', linestyle='--', label="Resonance Frequencies")
            # Label the vertical line
            plt.text(resfreq + 0.4 * self.basefreq, y_top * 0.02 , f"{modenum}", rotation=0, fontsize=7, color='red', ha='center')
            modenum = modenum + 1
            resfreq = resfreq + self.basefreq

        plt.xlabel("Frequency (Hz)")  # X-axis label
        plt.ylabel("Amplitude")  # Y-axis label
        plt.title(tname)  # Title
        plt.legend(handles = [peak_proxy])  # Show legend
        plt.grid(True)  # Add grid for better readability

        # Save the figure
        plt.savefig(Spectrum.picsdir + f"{figname}.png", dpi=300)  # Save as PNG with high resolution
        if showfig:
            plt.show()  # Show the plot


spec1 = Spectrum("Exp4-1-7-8cells-50mm-iris-16mm.dat", 50, ampthrs = 1)
spec1.plot_spectrum("Exp4-1-7-8cells-50mm-iris-16mm-Spectrum", "8x50mm cells + 7x16mm irises spectrum", annotate = True, showfig = False)
# spec1.dbg()
# print(spec1.peaks())
spec2 = Spectrum("Exp4-1-7-8cells-75mm-iris-16mm.dat", 75, ampthrs = 1)
spec2.plot_spectrum("Exp4-1-7-8cells-75mm-iris-16mm-Spectrum", "8x75mm cells + 7x16mm irises spectrum", annotate = True, showfig = True)










