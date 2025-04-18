# Third party packages
import math
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from dataclasses import dataclass, field
import numpy as np

matplotlib.use("Qt5Agg")  # Force the Qt backend

class InvalidLineFormatError(Exception):
    """Exception raised when a line does not contain exactly two floats."""
    
    def __init__(self, line, tokens):
        self.line = line
        self.tokens = tokens
        super().__init__(f"Invalid line format: {line!r} (parsed: {tokens})")

@dataclass
class Spectrum:
    """Manipulates a spectrum data. 

    Args:
        fname (str): Target data file name relative to the data directory "../data/". 
        tubelen (float): Tube length (mm). 
        unum (int): Number of unit cells, e.g., number of 5/7.5 mm cells. 
        wsize (int, optional): Peak windowing size. Defaults to 5. 
        ampthrs: (int, optional): Frequency peak threshold amplitude. Defaults 
            to None. 
    """
    # target data file name relative to the data directory "../data/"
    fname: str
    # tube length (mm)
    tubelen: float
    # number of unit cells, e.g. number of 5/7.5 mm tubes
    unum: int
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
    specplot: Figure = field(init = False)
    specax: Axes = field(init = False)
    dispplot: Figure = field(init = False)
    dispax: Axes = field(init = False)
    DOSplot: Figure = field(init = False)
    DOSax: Axes = field(init = False)
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
        """Parses a line with two floats.

        Args:
            line (str): Line to be parsed. 

        Returns:
            tuple(float, float): Two float data points parsed. 

        Raises:
            InvalidLineFormatError: If line does not consist of exactly two floats. 
        """
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

    def debug(self):
        """Print first 5 sets of data."""
        print(self.freq[0:5], self.amp[0:5])

    def peaks(self) -> list[tuple[float, float]]:
        """Return frequency peaks of this spectrum."""
        return self.pklist

    def plot_spectrum(self, figname: str, tname: str, annotate: bool = False, 
                      mark: bool = True, showfig: bool = True):
        """Plot the amplitude vs frequency spectrum.

        Args: 
            figname (str): The figure stored will be named as `{figname}.png`.
            tname (str): Plot title name. 
            annotate (bool, optional): To Annotate the peak frequencies or not. 
                Defaults to False. 
            showfig (bool, optional): Show the figure in an additional window. 
                Defaults to True. 
        """
        # Create the plot
        self.specplot, self.specax= plt.subplots(figsize=(10, 5))  # Set figure size
        self.specax.plot(self.freq, self.amp, label="Amplitude vs Frequency")  # Plot data

        x_bot, x_top = self.specax.get_xlim()
        x_len = x_top - x_bot
        y_bot, y_top = self.specax.get_ylim()
        y_len = y_top - y_bot

        # Annotate peaks
        if annotate:
            for peak in self.pklist:
                # Note that the precision of the spectrum software is 1 decimal place
                self.specax.annotate(f"{peak[0]:.1f} Hz", 
                             peak, 
                             textcoords="offset points",
                             xytext=(0, 10), 
                             ha='center', 
                             fontsize=9, 
                             color='red')

        # Mark peaks with vertical lines
        modenum = 1
        resfreq = self.basefreq

        # Mark original resonance frequencies
        if mark:
            while(modenum * self.basefreq <= self.freq[-1]):
                self.specax.axvline(x = resfreq, color='orange', linestyle='--', 
                                    alpha=0.7)
                # Define a proxy line (doesn't appear in plot, only legend)
                peak_proxy = mlines.Line2D(
                        [], [], 
                        color='orange', 
                        linestyle='--', 
                        label="Original Resonance Frequencies"
                        )
                # Label the vertical line
                self.specax.text(
                        resfreq + 0.4 * self.basefreq, 
                        y_top * 0.02 , 
                        f"{modenum}", 
                        rotation=0, 
                        fontsize=7, 
                        color='red', 
                        ha='center'
                        )
                modenum = modenum + 1
                resfreq = resfreq + self.basefreq
            self.specax.legend(handles = [peak_proxy])  # Show legend
        else:
            self.specax.legend()  # Show legend

        self.specax.set_xlabel("Frequency (Hz)")  # X-axis label
        self.specax.set_ylabel("Amplitude")  # Y-axis label
        self.specax.set_title(tname)  # Title
        self.specax.grid(True)  # Add grid for better readability

        # Save the figure
        self.specplot.savefig(Spectrum.picsdir + f"{figname}.png", dpi=300)  # Save as PNG with high resolution
        if showfig:
            plt.show()  # Show the plot
        plt.close(self.specplot) # Close so that the plot is not displayed further

    @staticmethod
    def adjust_freqs(removed_freqs: list[float], all_freqs: list[float],
                     ignore_freqs: list[float], add_freqs: list[float], allowed_freqs: int):
        """Remove ignored frequencies from auto-detected peak frequencies.

        Args: 
            removed_freqs (list[float]): Final resonance frequency array removed 
                specified frequencies.
            all_freqs (list[float]): Original detected resonance frequency array.
            ignore_freqs (list[float]): Frequencies to ignore.
            allowed_freqs (int): Maximum number of resonance frequencies to plot.
        """

        # Manually add frequencies into all_freqs array
        all_freqs += add_freqs 
        all_freqs.sort()

        i, j = 0, 0
        while i < len(all_freqs) and j < len(ignore_freqs) and len(removed_freqs) < allowed_freqs:
            if all_freqs[i] < ignore_freqs[j]:
                removed_freqs.append(all_freqs[i])
                i += 1
            elif ignore_freqs[j] < all_freqs[i]:
                j += 1
            else:
                i += 1
                j += 1
        
        while i < len(all_freqs) and len(removed_freqs) < allowed_freqs:
            removed_freqs.append(all_freqs[i])
            i += 1

    def plot_dispersion_relation(self, figname: str, tname: str, 
             showfig: bool = True, allowed_freqs: int = 20, 
             ignore_freqs: list[float] | None = None,
             add_freqs: list[float] | None = None):
        """Plot resonance frequencies f to wave numbers k_n.

        Note that the 0-th order resonance appear at k = 0 which is not observable. 
        Thus we start at n = 1 with the base wave number. 

        Args: 
            figname (str): The figure stored will be named as `{figname}.png`.
            tname (str): Plot title name. 
            showfig (bool, optional): Show the figure in an additional window. 
                Defaults to True. 
            allowed_freqs (int, optional): Maximum number of frequencies allowed. 
                Defaults to 20. 
            ignore_freqs (list[float], optional): List of frequencies to ignore. 
                Defaults to None. 
        """
        # Initialize ignored and added frequencies
        if ignore_freqs is None:
            ignore_freqs = []
        if add_freqs is None:
            add_freqs = []

        # Resonance frequency array and wave number array used in plot
        # Additionally added n = 0 resonance (freq, wavenum) = (0, 0)
        # ((n + self.unum) % (2 * self.unum) - self.unum): keeps wavenum in 
        # first Brillouin zone 
        rwavenum = [((n + self.unum) % (2 * self.unum) - self.unum) 
                    * self.basewavenum for n in range(0, allowed_freqs + 1)]
        rfreq = []
        Spectrum.adjust_freqs(rfreq, [x[0] for x in self.pklist], 
                ignore_freqs, add_freqs, allowed_freqs)
        rfreq = [0.0] + rfreq

        # Free space dispersion line
        ffreq = np.linspace(0, (allowed_freqs + 1) * self.basefreq, 500)
        # No irises resonance frequencies
        fresfreq = [n * self.basefreq for n in range(0, allowed_freqs + 1)]
        fwavenum = np.linspace(0, (allowed_freqs + 1) * self.basewavenum, 500)
        # Again keep wavenum in first Brillouim zone
        halfbzlen = self.unum * self.basewavenum
        fwavenum = (fwavenum + halfbzlen) % (2 * halfbzlen) - halfbzlen

        # Construct figure
        self.dispplot, self.dispax = plt.subplots(figsize=(10, 5))  

        # Plot data
        self.dispax.scatter(rwavenum, rfreq, label="Resonance Frequencies", 
                c = "orange")
        self.dispax.scatter(fwavenum, ffreq, s = 10, 
                            label="Free Space Dispersion Relation") 
        self.dispax.scatter(rwavenum, fresfreq, s = 30, 
                            label="Resonance Frequencies (No Irises)", c = "red") 

        # Draw freuquency (energy) bands
        l = 0
        while l < len(rfreq):
            r = min(l + self.unum - 1, len(rfreq) - 1)
            if l == 0:
                self.dispax.axhspan(rfreq[l], rfreq[r], color="orange",
                                    alpha=0.3, hatch="/", label = "Frequency bands")
            else:
                self.dispax.axhspan(rfreq[l], rfreq[r], color="orange",
                                    alpha=0.3, hatch="/")
            l += self.unum

        # Draw first Brillouin zone
        # self.dispax.axvspan(-halfbzlen, halfbzlen, color="green",
        #                     alpha=0.3, hatch="\\", label="First Brillouin Zone")
        self.dispax.axvline(x = halfbzlen, color='green', linestyle='--', 
                            alpha=0.7, label = "First Brillouin Zone Boundary")
        self.dispax.axvline(x = -halfbzlen, color='green', linestyle='--', alpha=0.7)

        # Plot information
        self.dispax.set_xlabel("Wave Number (1/mm)")  # X-axis label
        self.dispax.set_ylabel("Frequency (Hz)")  # Y-axis label
        self.dispax.set_title(tname)  # Title
        self.dispax.legend()  # Show legend
        self.dispax.grid(True)  # Add grid for better readability

        # Save the figure
        self.dispplot.savefig(Spectrum.picsdir + f"{figname}.png", dpi=300)  # Save as PNG with high resolution
        if showfig:
            plt.show()  # Show the plot
        plt.close(self.dispplot) # Close so that the plot is not displayed further

    def plot_DOS(self, figname: str, tname: str, showfig: bool = True, 
            allowed_freqs: int = 20, ignore_freqs: list[float] | None = None, 
            add_freqs: list[float] | None = None):
        """Plot the DOS D(f_i) w.r.t the resonance frequencies f_i. 

        DOS formula: 
            D(f_i) = 1 / (f_{i + 1} - f_i)

        Args: 
            figname (str): The figure stored will be named as `{figname}.png`.
            tname (str): Plot title name. 
            showfig (bool, optional): Show the figure in an additional window. 
                Defaults to True. 
            allowed_freqs (int, optional): Maximum number of frequencies allowed. 
                Defaults to 20. 
            ignore_freqs (list[float], optional): List of frequencies to ignore. 
                Defaults to None. 
        """
        # Initialize ignored and added frequencies
        if ignore_freqs is None:
            ignore_freqs = []
        if add_freqs is None:
            add_freqs = []

        # Resonance frequency array and wave number array used in plot
        rfreq = []
        Spectrum.adjust_freqs(rfreq, [x[0] for x in self.pklist], 
                ignore_freqs, add_freqs, allowed_freqs)

        # DOS calculation
        DOS = [1 / (rfreq[i + 1] - rfreq[i]) for i in range(0, len(rfreq) - 1)]

        self.DOSplot, self.DOSax = plt.subplots(figsize = (10, 5))
        self.DOSax.scatter(rfreq[:-1], DOS, label = "Density of States")

        self.DOSax.set_xlabel("Frequency (Hz)")  # X-axis label
        self.DOSax.set_ylabel("Density of States (s)")  # Y-axis label
        self.DOSax.set_title(tname)  # Title
        self.DOSax.legend()  # Show legend
        self.DOSax.grid(True)  # Add grid for better readability

        # Save the figure
        self.DOSplot.savefig(Spectrum.picsdir + f"{figname}.png", dpi=300)  # Save as PNG with high resolution
        if showfig:
            plt.show()  # Show the plot
        plt.close(self.DOSplot) # Close so that the plot is not displayed further


spec1 = Spectrum("Exp4-1-7-8cells-50mm-iris-16mm.dat", tubelen = 50, unum = 8, ampthrs = 1)
spec1.plot_spectrum("Exp4-1-7-8cells-50mm-iris-16mm-Spectrum", 
        "8x50mm Cells + 7x16mm Irises Spectrum", annotate = True, 
        showfig = False)
spec1.plot_dispersion_relation(
        "Exp4-1-7-8cells-50mm-iris-16mm-Dispersion-Relation", 
        "8x50mm Cells + 7x16mm Irises Dispersion Relation Plot", 
        ignore_freqs = [380.0], showfig = False)
spec1.plot_DOS("Exp4-1-7-8cells-50mm-iris-16mm-DOS", 
        "8x50mm Cells + 7x16mm Irises Density of States Plot", 
        ignore_freqs = [380.0], showfig = False, allowed_freqs = 20)
# spec1.debug()
# print(spec1.peaks())

spec2 = Spectrum("Exp4-1-7-8cells-75mm-iris-16mm.dat", tubelen = 75, unum = 8, ampthrs = 1)
spec2.plot_spectrum( "Exp4-1-7-8cells-75mm-iris-16mm-Spectrum", 
        "8x75mm Cells + 7x16mm Irises Spectrum", annotate = True, 
        showfig = False)
spec2.plot_dispersion_relation(
        "Exp4-1-7-8cells-75mm-iris-16mm-Dispersion-Relation", 
        "8x75mm Cells + 7x16mm Irises Dispersion Relation Plot", 
        showfig = False)

spec3 = Spectrum("Exp4-1-8-10cells-50mm-iris-16mm.dat", tubelen = 50, unum = 10, ampthrs = 1)
spec3.plot_spectrum( "Exp4-1-8-10cells-50mm-iris-16mm-Spectrum", 
        "10x50mm Cells + 9x16mm Irises Spectrum", annotate = True, 
        showfig = False)
spec3.plot_dispersion_relation(
        "Exp4-1-8-10cells-50mm-iris-16mm-Dispersion-Relation", 
        "10x50mm Cells + 9x16mm Irises Dispersion Relation Plot", 
        showfig = False, allowed_freqs = 25, add_freqs = [5190.0])

spec4 = Spectrum("Exp4-1-8-12cells-50mm-iris-16mm.dat", tubelen = 50, unum = 12, ampthrs = 1)
spec4.plot_spectrum( "Exp4-1-8-12cells-50mm-iris-16mm-Spectrum", 
        "12x50mm Cells + 11x16mm Irises Spectrum", annotate = True, 
        showfig = False)
spec4.plot_dispersion_relation(
        "Exp4-1-8-12cells-50mm-iris-16mm-Dispersion-Relation", 
        "12x50mm Cells + 11x16mm Irises Dispersion Relation Plot", 
        ignore_freqs = [130.0], showfig = False, allowed_freqs = 30, add_freqs = [5200.0])

spec5 = Spectrum("Exp4-1-9-8cells-50mm-iris-10mm.dat", tubelen = 50, unum = 8, ampthrs = 1)
spec5.plot_spectrum( "Exp4-1-9-8cells-50mm-iris-10mm-Spectrum", 
        "8x50mm Cells + 7x10mm Irises Spectrum", annotate = True, 
        showfig = False)
spec5.plot_dispersion_relation(
        "Exp4-1-9-8cells-50mm-iris-10mm-Dispersion-Relation", 
        "8x50mm Cells + 7x10mm Irises Dispersion Relation Plot", 
        ignore_freqs = [210.0], showfig = False, allowed_freqs = 20, add_freqs = [4300.0])

spec6 = Spectrum("Exp4-1-9-8cells-50mm-iris-13mm.dat", tubelen = 50, unum = 8, ampthrs = 1)
spec6.plot_spectrum( "Exp4-1-9-8cells-50mm-iris-13mm-Spectrum", 
        "8x50mm Cells + 7x13mm Irises Spectrum", annotate = True, 
        showfig = False)
spec6.plot_dispersion_relation(
        "Exp4-1-9-8cells-50mm-iris-13mm-Dispersion-Relation", 
        "8x50mm Cells + 7x13mm Irises Dispersion Relation Plot", 
        ignore_freqs = [350.0], showfig = False, allowed_freqs = 20)

spec7 = Spectrum("Exp4-3-1-all-13mm.dat", tubelen = 50, unum = 12, ampthrs = 1)
spec7.plot_spectrum( "Exp4-3-1-all-13mm-Spectrum", 
        "12x50mm Cells + 11x13mm Irises Spectrum", annotate = True, 
        showfig = False)
spec7.plot_dispersion_relation(
        "Exp4-3-1-all-13mm-Dispersion-Relation", 
        "12x50mm Cells + 11x13mm Irises Dispersion Relation Plot", 
        ignore_freqs = [780.0, 1010.0], add_freqs = [3380.0, 4760.0], 
        showfig = False, allowed_freqs = 28)

spec8 = Spectrum("Exp4-3-1-16mm-13mm.dat", tubelen = 50, unum = 12, ampthrs = 1)
spec8.plot_spectrum( "Exp4-3-1-16mm-13mm-Spectrum", 
        "12x50mm Cells + 6x13, 5x16mm Irises Spectrum", annotate = True, 
        showfig = False)
spec8.plot_dispersion_relation(
        "Exp4-3-1-16mm-13mm-Dispersion-Relation", 
        "12x50mm Cells + 6x13, 5x16mm Irises Dispersion Relation Plot", 
        ignore_freqs = [830.0], add_freqs = [4990.0], 
        showfig = False, allowed_freqs = 28)

spec9 = Spectrum("Exp4-3-2-spectrum.dat", tubelen = 125, unum = 5, ampthrs = 1)
spec9.plot_spectrum( "Exp4-3-2-spectrum-Spectrum", 
        "5x(50mm Cell + 16mm Iris + 75mm Cell) Spectrum", annotate = True, 
        mark = False, showfig = False)
spec9.plot_dispersion_relation(
        "Exp4-3-2-spectrum-Dispersion-Relation", 
        "5x(50mm Cell + 16mm Iris + 75mm Cell) Dispersion Relation Plot", 
        ignore_freqs = [], add_freqs = [], 
        showfig = False, allowed_freqs = 30)

spec10 = Spectrum("Exp4-4-1-nodefect.dat", tubelen = 50, unum = 12, ampthrs = 1)
spec10.plot_spectrum( "Exp4-4-1-nodefect-Spectrum", 
        "12x50mm Cell + 11x16mm Iris Spectrum", annotate = True, 
        mark = False, showfig = False)
spec10.plot_dispersion_relation(
        "Exp4-4-1-nodefect-Dispersion-Relation", 
        "12x50mm Cell + 11x16mm Iris Dispersion Relation Plot", 
        ignore_freqs = [], add_freqs = [5210.0], 
        showfig = False, allowed_freqs = 30)

spec11 = Spectrum("Exp4-4-1-defect-head.dat", tubelen = 50, unum = 12, ampthrs = 1)
spec11.plot_spectrum( "Exp4-4-1-defect-head-Spectrum", 
        "11x50mm Cell + 11x16mm Iris + 1x75mm defect in front Spectrum", annotate = True, 
        mark = False, showfig = False)
spec11.plot_dispersion_relation(
        "Exp4-4-1-defect-head-Dispersion-Relation", 
        "11x50mm Cell + 11x16mm Iris + 1x75mm defect in front Spectrum", 
        ignore_freqs = [], add_freqs = [], 
        showfig = False, allowed_freqs = 30)

def plot_analog_dispersion_relation(figname: str, tname: str, wavenumrange: float = 100, 
                                    halfzonewidth: float = 20, 
                                    a: float = 1, b: float = 0.01, 
                                    showfig: bool = True):
    # wave number array
    wavenum = np.linspace(- wavenumrange, wavenumrange, 1000)
    # reduced zone wave number array
    redwavenum = ((wavenum + halfzonewidth) % (2 * halfzonewidth) - halfzonewidth)
    # Sound wave frequency (dispersion relation), a = c/2L
    sfreq = a * wavenum
    # Electron energy (dispersion relation), b = \hbar^2/2m
    energy = b * wavenum**2
    # Create the plot
    fig, axs = plt.subplots(figsize=(10, 5))  # Set figure size
    axs.scatter(wavenum, sfreq, s = 10, label="Sound Wave Dispersion Relation") 
    axs.scatter(redwavenum, sfreq, s = 10, label="Sound Wave Dispersion Relation (Reduced Zone)") 
    axs.scatter(wavenum, energy, s = 10, label="Electron Dispersion Relation") 
    axs.scatter(redwavenum, energy, s = 10, label="Electron Dispersion Relation (Reduced Zone)") 

    zonenum = int(wavenumrange // halfzonewidth)
    for n in [i for i in range(-zonenum, zonenum + 1) if i % 2 == 1]:
        if n == 1:
            axs.axvline(x = n * halfzonewidth, color='orange', linestyle='--',
                        label = "Brillouin Zone Boundary")
        else:
            axs.axvline(x = n * halfzonewidth, color='orange', linestyle='--')

    axs.set_xlabel("Wave Number")  # X-axis label
    axs.set_ylabel("Frequency (Sound), Energy (Electron)")  # Y-axis label
    axs.set_title(tname)  # Title
    axs.legend()  # Show legend
    axs.grid(True)  # Add grid for better readability
    # Save the figure
    fig.savefig(Spectrum.picsdir + f"{figname}.png", dpi=300)  # Save as PNG with high resolution
    if showfig:
        plt.show()  # Show the plot
    plt.close(fig) # Close so that the plot is not displayed further

plot_analog_dispersion_relation("Exp4-1-Dispersion-Relation-Analog", 
                                "Dispersion Relation Analog between Sound and Electron", 
                                showfig = False)
