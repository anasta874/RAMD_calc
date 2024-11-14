import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Загрузка данных из Excel
def load_excel(file_path, skip_rows=2):
    data = pd.read_excel(file_path, header=None).to_numpy()
    return data[skip_rows:, :]

# Фильтрация данных
def filter_data(force_data):
    valid_data = (force_data[:, 1] > 0) & (~np.isnan(force_data[:, 1]))
    t = force_data[valid_data, 0]  # Время
    N = force_data[valid_data, 1]  # Значения N
    return t, N

# Функция для расчета tau и lambda
def calculate_dissociation_rates(t, n, N, N_cut_max, tau_frac, tau_var, lambda_var, pend_var):
    for N_cut in range(N_cut_max):
        # Фильтрация данных для линейного приближения
        t_fit = t[(N > N_cut) & (t > tau_frac)]
        n_fit = n[(N > N_cut) & (t > tau_frac)]

        if len(t_fit) > 1:  # Проверка, достаточно ли данных для приближения
            # Линейное приближение
            pend = np.polyfit(t_fit, n_fit, 1)
            tau = round((1 / abs(pend[0])) / 1e4, 15)  # Масштабирование tau
            lambda_ = round(pend[0] * 1e4, 15)         # Масштабирование lambda

            # Сохранение результатов в списки
            tau_var.append(tau)
            lambda_var.append(lambda_)
            pend_var.append(pend)

# Функция для построения графика с выделенными линиями для N_cut и N_cut_max
def plot_tau_with_cutoff(x_values, y_values, N_cut, N_cut_max, force_value, output_file=None):
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, y_values, label=r'$\tau$ values', marker='o', color='blue')
    vline_cut = plt.axvline(x=N_cut, color='red', linestyle='--', linewidth=2, label=r'$N_{cut}$')
    vline_cut_max = plt.axvline(x=N_cut_max, color='green', linestyle='--', linewidth=2, label=r'$N_{cut\ max}$')
    plt.legend(loc='best')
    plt.title(f"Force {force_value}", fontsize=18)
    plt.ylabel(r'$\tau$, ps', fontsize=18)
    plt.xlabel(r'$N_{cut}$', fontsize=18)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1e4:.1f}'))
    plt.text(0.5, max(y_values) * 1.05, r'$ \times 10^4$', fontsize=16, color='black', verticalalignment='bottom', horizontalalignment='center')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.show()

# Plot the initial linear region in logarithmic coordinates
def plot_init_part_ln_ln(force_value, n, t, t0, tend, output_file=None):
    # Filter positive values in the range t0:tend
    valid_indices = (t[t0:tend] > 0) & (n[t0:tend] < 0)
    filtered_t = t[t0:tend][valid_indices]
    filtered_n = n[t0:tend][valid_indices]
    pinit = np.polyfit(np.log(filtered_t), np.log(-filtered_n), 1)
    p = pinit[0]
    
    # plt.figure()
    # plt.plot(np.log(filtered_t), np.log(-filtered_n), 'o', color='magenta', linewidth=1.5, label='Data Points')
    # plt.plot(np.log(filtered_t), np.polyval(pinit, np.log(filtered_t)), '--', color='red', linewidth=1.5, label='Fitted Line')
    # plt.xlabel('ln(t/ps)', fontsize=14)
    # plt.ylabel('ln(ln(N/N_0))', fontsize=14)
    # plt.title(f'Force Value: {force_value}', fontsize=16)
    # plt.legend()

    # if output_file:
    #     plt.savefig(output_file)
    # plt.show()
    return p, pinit

# Function to plot ln(N(t)/N(0))
def plot_lnN(force_value, t, t_fit, n, tau_frac, pend, pinit, output_file=None):
    plt.figure()
    plt.plot(t, n, 'o', color='magenta', linewidth=1.5, label='Data')
    plt.plot(t_fit, np.polyval(pend, t_fit), color='blue', linewidth=1.5, label='Fitted Line')
    # Filter positive values for the logarithm
    valid_indices = (t > 0) & (t < tau_frac)
    valid_t = t[valid_indices]
    # Plot the exponential region with filtering
    plt.plot(valid_t, -np.exp(np.polyval(pinit, np.log(valid_t))), '--', color='black', linewidth=1.5, label='Exponential Fit')
    plt.axvline(x=tau_frac, linestyle='--', color='red')
    plt.ylim([1.1 * np.min(n), 0.1])
    plt.xlabel(r'$t, \mathrm{ps}$', fontsize=18)
    plt.ylabel(r'$\ln(N(t)/N(0))$', fontsize=18)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/10000:.1f}'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Force Value: {force_value}', fontsize=10)
    plt.legend()
    if output_file:
        plt.savefig(output_file)
    plt.show()

# Function to plot N(t)
def plot_N(force_value, t, t_fit, N, N0, tau_frac, pend, pinit, output_file=None):
    t_fit = np.array(t_fit)
    plt.figure()
    plt.plot(t, N, 'o', color='magenta', linewidth=1.5, label='Data')
    plt.plot(t[t < tau_frac], N[0] * np.exp(-np.exp(pinit[1]) * t[t < tau_frac] ** pinit[0]), '--', color='black', linewidth=1.5, label='Exponential Fit')
    plt.plot(t_fit, N0 * np.exp(pend[1]) * np.exp(pend[0] * t_fit), color='blue', linewidth=1.5, label='Fitted Line')
    plt.plot([1 / abs(pend[0]), 1 / abs(pend[0])], [0, N0], '-.', color='black')
    plt.axvline(x=tau_frac, linestyle='--', color='red')
    plt.xlabel(r'$t, \mathrm{ps}$', fontsize=18)
    plt.ylabel(r'$N(t)$', fontsize=18)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/10000:.1f}'))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Force Value: {force_value}', fontsize=10)
    plt.legend()
    if output_file:
        plt.savefig(output_file)
    plt.show()
