import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

# Загрузка данных из Excel
def load_excel(file_path, skip_rows=2):
    data = pd.read_excel(file_path, header=None).to_numpy()
    return data[skip_rows:, :]

# Фильтрация данных
def filter_data(time, values, n, N_cut, tau_frac):
    t_fit, n_fit = [], []
    for i in range(len(time)):
        if values[i] > N_cut and time[i] > tau_frac:
            t_fit.append(time[i])
            n_fit.append(n[i]) 
    return t_fit, n_fit

# Построение графика с выделением точки обрезки
def plot_tau_with_cutoff(x_values, y_values, N_cut, tau_var, force_value, output_file=None):
    plt.figure()
    plt.plot(x_values, y_values, label=r'$\tau$ values')
    vline = plt.axvline(x=N_cut, color='red', linestyle='--', linewidth=2)
    plt.legend([vline], ['Cropping point'], loc='best')
    plt.title(f"Force {force_value}", fontsize=18)
    plt.ylabel(r'$\tau$, ps', fontsize=18)
    plt.xlabel(r'$N_{cut}$', fontsize=18)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1e4:.1f}'))
    plt.text(0.5, max(y_values)*1.05, r'$ \times 10^4$', fontsize=16, color='black', verticalalignment='bottom', horizontalalignment='center')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    if output_file:
        plt.savefig(output_file)
    plt.show()

# Построение начального линейного участка в логарифмических координатах
def plot_init_part_ln_ln(force_value, n, t, t0, tend, output_file=None):
    # Фильтрация положительных значений в диапазоне t0:tend
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


# Функция для построения графика ln(N(t)/N(0))
def plot_lnN(force_value, t, t_fit, n, tau_frac, pend, pinit, output_file=None):
    plt.figure()
    plt.plot(t, n, 'o', color='magenta', linewidth=1.5, label='Data')
    plt.plot(t_fit, np.polyval(pend, t_fit), color='blue', linewidth=1.5, label='Fitted Line')
    # Фильтрация положительных значений для логарифма
    valid_indices = (t > 0) & (t < tau_frac)
    valid_t = t[valid_indices]
    # Построение графика экспоненциального участка с фильтрацией
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


# Функция для построения графика N(t)
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

#  Построение двух графиков в одном:
def two_in_one(force_value, t, t_fit, n, N, N0, tau_frac, pend, pinit, xlim1, ylim1, xlim2, axs, output_file='output_two_in_one.png'):
    pend, pinit, t_fit = np.array(pend), np.array(pinit), np.array(t_fit)
    fig, ax1 = plt.subplots(figsize=(7, 5)) 
    plt.subplots_adjust(left=0.15, right=0.9, top=0.85, bottom=0.15) 
    ax1.plot(t, n, 'o', color='magenta', markersize=3, linewidth=1.5)
    if len(t[t < tau_frac]) > 0:
        ax1.plot(t_fit, np.polyval(pend, t_fit), color='blue', linewidth=1)
        ax1.plot(t[t < tau_frac], -np.exp(np.polyval(pinit, np.log(t[t < tau_frac]))), '--', color='black', linewidth=1)
    ax1.set(xlim=xlim1, ylim=ylim1, xlabel=r'$t, \mathrm{ps}$', ylabel=r'$\ln(N(t)/N(0))$')
    ax1.tick_params(axis='both', labelsize=10)
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*1e-5:.1f}'))  # Оси X без тысяч
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Построение второго графика: N(t) vs t (в маленьком окошке)
    ax2 = fig.add_axes([0.55, 0.45, 0.4, 0.4])
    ax2.plot(t, N, 'o', color='magenta', markersize=3, linewidth=1.5)
    if len(t[t < tau_frac]) > 0:
        ax2.plot(t[t < tau_frac], N[0] * np.exp(-np.exp(pinit[1]) * t[t < tau_frac] ** pinit[0]), '--', color='black', linewidth=1)
    ax2.plot(t_fit, N0 * np.exp(pend[1]) * np.exp(pend[0] * t_fit), color='blue', linewidth=1)
    ax2.axvline(x=tau_frac, linestyle='--', color='red', linewidth=1)
    ax2.set(xlim=xlim2, ylim=[0, N0 * 1.2], xlabel=r'$t, \mathrm{ps}$', ylabel=r'$N(t)$')
    ax2.tick_params(axis='both', labelsize=8)
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*1e-5:.1f}'))
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
