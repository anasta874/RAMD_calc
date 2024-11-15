# Stat_func.py

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from math import exp
import matplotlib.pyplot as plt

# Define linear model
def linear_model(x, a, b):
    return a * x + b

# Linear fitting and confidence intervals
def fit_with_intervals(f, log_log_tau_calc, confidence=0.68):
    popt, pcov = curve_fit(linear_model, f, log_log_tau_calc)
    a, b = popt
    ci = np.sqrt(np.diag(pcov)) * t.ppf(confidence, len(f) - 2)
    predicted_log_log_tau = linear_model(f, a, b)
    pred_interval = 1.96 * np.std(log_log_tau_calc - predicted_log_log_tau)
    return a, b, ci, predicted_log_log_tau, pred_interval

# Exclude outliers and refit
def exclude_outliers_and_refit(f, log_log_tau_calc, predicted_log_log_tau, pred_interval):
    mask = (log_log_tau_calc > predicted_log_log_tau - pred_interval) & (log_log_tau_calc < predicted_log_log_tau + pred_interval)
    filtered_f = f[mask]
    filtered_log_log_tau = log_log_tau_calc[mask]
    popt_filtered, pcov_filtered = curve_fit(linear_model, filtered_f, filtered_log_log_tau)
    a_filtered, b_filtered = popt_filtered
    pred_interval_filtered = 1.96 * np.std(filtered_log_log_tau - linear_model(filtered_f, a_filtered, b_filtered))
    return a_filtered, b_filtered, filtered_f, filtered_log_log_tau, pred_interval_filtered

# Calculate dissociation time at f=0
def calculate_dissociation_time(b):
    return exp(exp(b)) * 1e-12

def plot_dissociation_data(f, log_log_tau_calc, ftau, a, b, predicted_log_log_tau, pred_interval,
                           a_filtered=None, b_filtered=None, pred_interval_filtered=None, FS=22):
    plt.figure(figsize=(8, 4))
    plt.plot(0, 3.4514, '*', color='magenta', markersize=15, label='Experimental Point')
    plt.plot(f, log_log_tau_calc, 'o', color='blue', linewidth=0.75, label='Data Points')
    plt.plot(ftau, linear_model(ftau, a, b), label='Fit (Original Data)', color='blue', linewidth=1.5)
    plt.fill_between(ftau, linear_model(ftau, a, b) - pred_interval, linear_model(ftau, a, b) + pred_interval,
                     color='black', alpha=0.2, linestyle='--', label='68% Prediction Interval')

    # Refit line for filtered data, if provided
    if a_filtered is not None and b_filtered is not None and pred_interval_filtered is not None:
        plt.plot(ftau, linear_model(ftau, a_filtered, b_filtered), '--', color='green', label='Fit (Filtered Data)')
        plt.fill_between(ftau, linear_model(ftau, a_filtered, b_filtered) - pred_interval_filtered,
                         linear_model(ftau, a_filtered, b_filtered) + pred_interval_filtered, color='green', alpha=0.2,
                         label='68% Prediction Interval (Filtered)')

    plt.xlim([-10, 650])
    plt.xticks(range(0, 651, 100))
    plt.xlabel(r'$\mathit{f},\,\mathrm{kJ/(nm \cdot mol)}$', fontsize=FS)
    plt.ylabel(r'$\mathit{\ln(\ln(\tau))},\, \mathrm{ps}$', fontsize=FS)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.title("Dissociation Time Analysis with Prediction Intervals", fontsize=14)
    plt.show()