import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

matplotlib.use("Qt5Agg")  # Force the Qt backend

def double_slit_intensity_with_width(theta, I0, d, b, wavelength):
    """
    Intensity pattern for a double slit with finite width (Fraunhofer diffraction).

    Args:
        theta (numpy.ndarray): Array of detector angles in radians.
        I0 (float): Central intensity (scaling factor).
        d (float): Slit separation distance.
        b (float): Slit width.
        wavelength (float): Wavelength of light.

    Returns:
        numpy.ndarray: Array of intensity values at the given angles.
    """
    beta = (np.pi * b * np.sin(theta)) / wavelength
    alpha = (np.pi * d * np.sin(theta)) / wavelength
    intensity = I0 * (np.cos(alpha))**2 * (np.sinc(beta / np.pi))**2
    return intensity

def double_slit_intensity_without_width(theta, I0, d, wavelength):
    """
    Intensity pattern for a double slit assuming negligible slit width.

    Args:
        theta (numpy.ndarray): Array of detector angles in radians.
        I0 (float): Central intensity (scaling factor).
        d (float): Slit separation distance.
        wavelength (float): Wavelength of light.

    Returns:
        numpy.ndarray: Array of intensity values at the given angles.
    """
    alpha = (np.pi * d * np.sin(theta)) / wavelength
    intensity = I0 * (np.cos(alpha))**2
    return intensity

def fit_double_slit(theta_data, amplitude_data, wavelength, use_width=True, initial_guess=None):
    """
    Fits the double-slit diffraction pattern to the given data.

    Args:
        theta_data (numpy.ndarray): Array of detector angles in radians.
        amplitude_data (numpy.ndarray): Array of detected amplitudes.
        wavelength (float): Wavelength of light.
        use_width (bool, optional): Whether to include the effect of slit width. Defaults to True.
        initial_guess (list or tuple, optional): Initial guess for the fitting parameters [I0, d, b] or [I0, d]. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - popt (numpy.ndarray): Optimal values for the fitting parameters.
            - pcov (numpy.ndarray): Covariance matrix of the estimated parameters.
            - fitted_intensity (numpy.ndarray): Fitted intensity values.
    """
    intensity_data = amplitude_data**2  # Intensity is proportional to amplitude squared

    if use_width:
        def func_with_width(theta, I0, d, b):
            return double_slit_intensity_with_width(theta, I0, d, b, wavelength)
        if initial_guess is None:
            initial_guess = [np.max(intensity_data), 1e-2, 1e-2] # Example initial guess
        popt, pcov = curve_fit(func_with_width, theta_data, intensity_data, p0=initial_guess)
        fitted_intensity = func_with_width(theta_data, *popt)
    else:
        def func(theta, I0, d):
            return double_slit_intensity_without_width(theta, I0, d, wavelength)
        if initial_guess is None:
            initial_guess = [np.max(intensity_data), 1e-2] # Example initial guess
        popt, pcov = curve_fit(func, theta_data, intensity_data, p0=initial_guess)
        fitted_intensity = func(theta_data, *popt)

    return popt, pcov, fitted_intensity

# --- Example Usage (Replace with your actual data) ---
# Simulate some data for demonstration
wavelength = 2.85e-2  # Example wavelength (Red laser)
d_true = 6e-2       # True slit separation
b_true = 1.5e-2        # True slit width
theta_true_data_deg = np.linspace(0, 80, 1000)
theta_true_data_rad = np.deg2rad(theta_true_data_deg)
intensity_true_with_width = double_slit_intensity_with_width(theta_true_data_rad, 1, d_true, b_true, wavelength)
# amplitude_data_with_noise = np.sqrt(intensity_true_with_width) + 0.1 * np.random.randn(len(theta_data_rad))

# Specify the path to your CSV file
file_path = '../data/MPL_A3-Exp6-Avg.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Extract the angle data and convert to radians
theta_data_deg_6cm = df["6cm-angle(deg)"].to_numpy()
theta_data_rad_6cm = np.deg2rad(theta_data_deg_6cm)
theta_data_deg_9cm = df["9cm-angle(deg)"].to_numpy()
theta_data_rad_9cm = np.deg2rad(theta_data_deg_9cm)

# Extract the amplitude data
amplitude_data_6cm = df["6cm-amp(mA)"].to_numpy()
amplitude_uct_6cm = df["6cm-amp-uct(mA)"].to_numpy()
amplitude_data_9cm = df["9cm-amp(mA)"].to_numpy()
amplitude_uct_9cm = df["9cm-amp-uct(mA)"].to_numpy()

# --- Fit with slit width ---
popt_with_width_6cm, pcov_with_width_6cm, fitted_intensity_with_width_6cm = fit_double_slit(
    theta_data_rad_6cm, amplitude_data_6cm, wavelength, use_width=True, initial_guess=[6.0, 6e-2, 1.5e-2]
)
popt_with_width_9cm, pcov_with_width_9cm, fitted_intensity_with_width_9cm = fit_double_slit(
    theta_data_rad_9cm, amplitude_data_9cm, wavelength, use_width=True, initial_guess=[6.0, 9e-2, 1.5e-2]
)
I0_fit_w_6cm, d_fit_w_6cm, b_fit_w_6cm = popt_with_width_6cm
amp_fit_w_curve_6cm = double_slit_intensity_with_width(theta_true_data_rad, I0_fit_w_6cm, d_fit_w_6cm, b_fit_w_6cm, wavelength)
I0_fit_w_9cm, d_fit_w_9cm, b_fit_w_9cm = popt_with_width_9cm
amp_fit_w_curve_9cm = double_slit_intensity_with_width(theta_true_data_rad, I0_fit_w_9cm, d_fit_w_9cm, b_fit_w_9cm, wavelength)
errors_w_6cm = np.sqrt(np.diag(pcov_with_width_6cm))
errors_w_9cm = np.sqrt(np.diag(pcov_with_width_9cm))
I0_err_w_6cm, d_err_w_6cm, b_err_w_6cm = errors_w_6cm
I0_err_w_9cm, d_err_w_9cm, b_err_w_9cm = errors_w_9cm
#
print("\n--- Fit with Slit Width ---")
print(f"Fitted I0 (6cm): {I0_fit_w_6cm:.3f} ± {I0_err_w_6cm:.3f}")
print(f"Fitted I0 (9cm): {I0_fit_w_9cm:.3f} ± {I0_err_w_9cm:.3f}")
print(f"Fitted d (6cm): {d_fit_w_6cm*1e2:.3f} cm ± {d_err_w_6cm*1e2:.3f} cm")
print(f"Fitted d (9cm): {d_fit_w_9cm*1e2:.3f} cm ± {d_err_w_9cm*1e2:.3f} cm")
print(f"Fitted b (6cm): {b_fit_w_6cm*1e2:.3f} cm ± {b_err_w_6cm*1e2:.3f} cm")
print(f"Fitted b (9cm): {b_fit_w_9cm*1e2:.3f} cm ± {b_err_w_9cm*1e2:.3f} cm")
#
# # --- Fit without slit width ---
popt_without_width_6cm, pcov_without_width_6cm, fitted_intensity_without_width_6cm = fit_double_slit(
    theta_data_rad_6cm, amplitude_data_6cm, wavelength, use_width=False, initial_guess=[6.0, 6e-2]
)
popt_without_width_9cm, pcov_without_width_9cm, fitted_intensity_without_width_9cm = fit_double_slit(
    theta_data_rad_9cm, amplitude_data_9cm, wavelength, use_width=False, initial_guess=[6.0, 9e-2]
)
I0_fit_wo_w_6cm, d_fit_wo_w_6cm = popt_without_width_6cm
amp_fit_wo_w_curve_6cm = double_slit_intensity_without_width(theta_true_data_rad, I0_fit_wo_w_6cm, d_fit_wo_w_6cm, wavelength)
I0_fit_wo_w_9cm, d_fit_wo_w_9cm = popt_without_width_9cm
amp_fit_wo_w_curve_9cm = double_slit_intensity_without_width(theta_true_data_rad, I0_fit_wo_w_9cm, d_fit_wo_w_9cm, wavelength)
errors_wo_w_6cm = np.sqrt(np.diag(pcov_without_width_6cm))
errors_wo_w_9cm = np.sqrt(np.diag(pcov_without_width_9cm))
I0_err_wo_w_6cm, d_err_wo_w_6cm = errors_wo_w_6cm
I0_err_wo_w_9cm, d_err_wo_w_9cm = errors_wo_w_9cm
#
print("\n--- Fit without Slit Width ---")
print(f"Fitted I0 (6cm): {I0_fit_wo_w_6cm:.3f} ± {I0_err_wo_w_6cm:.3f}")
print(f"Fitted I0 (9cm): {I0_fit_wo_w_9cm:.3f} ± {I0_err_wo_w_9cm:.3f}")
print(f"Fitted d (6cm): {d_fit_wo_w_6cm*1e2:.3f} cm ± {d_err_wo_w_6cm*1e2:.3f} cm")
print(f"Fitted d (9cm): {d_fit_wo_w_9cm*1e2:.3f} cm ± {d_err_wo_w_9cm*1e2:.3f} cm")
#
# # --- Plotting ---
fig, axs = plt.subplots(figsize=(10, 6))
# axs.scatter(np.degrees(theta_data_rad_6cm), amplitude_data_6cm, label='Experimental Amplitude (6cm)', s=30)
axs.errorbar(np.degrees(theta_data_rad_6cm), amplitude_data_6cm, yerr=amplitude_uct_6cm, fmt='o', markersize=5, capsize=3, label='Experimental Amplitude (6cm)')
# axs.scatter(np.degrees(theta_data_rad_9cm), amplitude_data_9cm, label='Experimental Amplitude (9cm)', s=30)
axs.errorbar(np.degrees(theta_data_rad_9cm), amplitude_data_9cm, yerr=amplitude_uct_9cm, fmt='o', markersize=5, capsize=3, label='Experimental Amplitude (6cm)')
# axs.plot(np.degrees(theta_data_rad_6cm), np.sqrt(fitted_intensity_with_width_6cm), label=f'Fit with Width (6cm, d={d_fit_w_6cm*1e6:.2f}µm, b={b_fit_w_6cm*1e6:.2f}µm)', color='red')
# axs.plot(np.degrees(theta_data_rad_9cm), np.sqrt(fitted_intensity_with_width_9cm), label=f'Fit with Width (9cm, d={d_fit_w_9cm*1e6:.2f}µm, b={b_fit_w_9cm*1e6:.2f}µm)', color='red')
# axs.plot(np.degrees(theta_data_rad_6cm), np.sqrt(fitted_intensity_without_width_6cm), label=f'Fit without Width (6cm, d={d_fit_wo_w_6cm*1e6:.2f}µm)', color='green', linestyle='--')
# axs.plot(np.degrees(theta_data_rad_9cm), np.sqrt(fitted_intensity_without_width_9cm), label=f'Fit without Width (9cm, d={d_fit_wo_w_9cm*1e6:.2f}µm)', color='green', linestyle='--')
axs.plot(np.degrees(theta_true_data_rad), np.sqrt(amp_fit_w_curve_6cm), label=f'Fit with Width (d=6cm, b=1.5cm, dfit={d_fit_w_6cm*1e2:.2f}cm, bfit={b_fit_w_6cm*1e2:.2f}cm)', color='blue')
axs.plot(np.degrees(theta_true_data_rad), np.sqrt(amp_fit_w_curve_9cm), label=f'Fit with Width (d=9cm, b=1.5cm, dfit={d_fit_w_9cm*1e2:.2f}cm, bfit={b_fit_w_9cm*1e2:.2f}cm)', color='orange')
axs.plot(np.degrees(theta_true_data_rad), np.sqrt(amp_fit_wo_w_curve_6cm), label=f'Fit without Width (d=6cm, dfit={d_fit_wo_w_6cm*1e2:.2f}cm)', color='blue', linestyle='--')
axs.plot(np.degrees(theta_true_data_rad), np.sqrt(amp_fit_wo_w_curve_9cm), label=f'Fit without Width (d=9cm, dfit={d_fit_wo_w_9cm*1e2:.2f}cm)', color='orange', linestyle='--')
#
axs.set_xlabel('Detector Angle θ (degrees)')
axs.set_ylabel('Amplitude')
axs.set_title('Double Slit Diffraction Pattern Fitting')
axs.legend()
axs.grid(True)
fig.tight_layout()
plt.show()

fig.savefig("../pics/Exp6-Fit.png", dpi=300)
