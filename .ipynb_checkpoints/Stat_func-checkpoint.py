# Stat_func.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.stats import t, probplot 
from math import exp
import scipy.stats as stats


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

def plot_filtered_dissociation_data(f, log_log_tau_calc, ftau, f_filtered, log_log_tau_filtered, 
                                    pred_interval_filtered, a_filtered, b_filtered, FS=22):
  
    plt.figure(figsize=(8, 4))
    plt.plot(0, 3.4514, '*', color='magenta', markersize=15, label='Experimental Point')
    plt.plot(f_filtered, log_log_tau_filtered, 'o', color='black', linewidth=0.5, label='Filtered Data Points')
    plt.plot(ftau, a_filtered * ftau + b_filtered, color='black', linewidth=1, label='Fit (Filtered Data)')
    plt.fill_between(ftau, (a_filtered * ftau + b_filtered) - pred_interval_filtered,
                     (a_filtered * ftau + b_filtered) + pred_interval_filtered,
                     color='black', alpha=0.2, linestyle='--', label='68% Prediction Interval (Filtered)')
    plt.xlim([-5, 650])
    plt.xlabel(r'$\mathit{f},\,\mathrm{kJ/(nm \cdot mol)}$', fontsize=FS)
    plt.ylabel(r'$\mathit{\ln(\ln(\tau))},\, \mathrm{ps}$', fontsize=FS)
    plt.xticks(range(0, 651, 100))
    plt.grid(True)
    plt.title("Filtered Dissociation Time Analysis with Prediction Intervals", fontsize=FS - 3)
    plt.legend(fontsize=10)
    plt.show()

def plot_residuals(f, rel_res, f_filtered, rel_res_filtered, FS=22):
    plt.figure(figsize=(10, 4))
    # Plot residuals for original data:
    plt.plot(f, rel_res, 'o', color='blue', linewidth=1.5, markersize=6, label='Original Residuals')
    # Plot residuals for filtered data (outliers removed):
    plt.plot(f_filtered, rel_res_filtered, 'o', color='black', linewidth=0.75, markersize=2.5, label='Filtered Residuals')
    plt.axhline(0, color='black', linestyle='-.', linewidth=0.75)
    plt.xlim([-20, 660])
    plt.xlabel(r'$\mathit{f},\,\mathrm{kJ/(nm \cdot mol)}$', fontsize=FS)
    plt.ylabel(r'$\mathrm{Rel.~residuals}$', fontsize=FS)
    plt.title("Residuals Analysis", fontsize=FS - 2)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.show()


def plot_qq(rel_res, rel_res_filtered, FS=22):
    plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(1, 2, 1) 
    stats.probplot(rel_res, dist="norm", plot=ax1)
    ax1.get_lines()[0].set_marker('o')
    ax1.get_lines()[0].set_markeredgecolor('blue')
    ax1.get_lines()[0].set_markersize(5)
    ax1.get_lines()[1].set_color('blue')
    ax1.get_lines()[1].set_linewidth(0.75)
    plt.title("Q-Q Plot of Original Residuals")
    plt.grid(True)

    ax2 = plt.subplot(1, 2, 2)
    stats.probplot(rel_res_filtered, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_marker('o')
    ax2.get_lines()[0].set_markeredgecolor('black')
    ax2.get_lines()[0].set_markersize(5)
    ax2.get_lines()[1].set_color('black')
    ax2.get_lines()[1].set_linewidth(0.75)
    plt.title("Q-Q Plot of Filtered Residuals")
    plt.grid(True)

    plt.suptitle("Q-Q Plot Analysis", fontsize=FS)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_qq2in1(rel_res, rel_res_filtered, FS=22):
    plt.figure(figsize=(8, 6)) 
    ax = plt.subplot(1, 1, 1) 

    probplot(rel_res, dist="norm", plot=ax)
    ax.get_lines()[0].set_marker('o')
    ax.get_lines()[0].set_markeredgecolor('blue')
    ax.get_lines()[0].set_markersize(5)
    ax.get_lines()[1].set_color('blue')
    ax.get_lines()[1].set_linewidth(0.75)

    probplot(rel_res_filtered, dist="norm", plot=ax)
    ax.get_lines()[2].set_marker('o')
    ax.get_lines()[2].set_markeredgecolor('black')
    ax.get_lines()[2].set_markersize(5)
    ax.get_lines()[3].set_color('black')
    ax.get_lines()[3].set_linewidth(0.75)

    plt.title("Q-Q Plot Analysis of Original and Filtered Residuals", fontsize=FS)
    plt.grid(True)
    plt.legend(['Original Residuals', 'Fit (Original)', 'Filtered Residuals', 'Fit (Filtered)'], fontsize=10)
    plt.show()