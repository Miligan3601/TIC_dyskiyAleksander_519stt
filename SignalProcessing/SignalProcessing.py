import numpy as np
import matplotlib.pyplot as plt


n = 500
Fs = 1000
F_max = 9
t = np.linspace(0, n/Fs, n, endpoint=False)
signal = np.sin(2 * np.pi * F_max * t) + 0.5 * np.sin(2 * np.pi * 2 * F_max * t) + 0.2 * np.random.normal(size=len(t))
filtered_signal = np.convolve(signal, np.ones(10)/10, mode='same')

signal_quantized = []
signal_dispersion = []
signal_snr = []

for M in [4, 16, 64, 256]:

    bits = []
    bit_signal = []

    delta = (np.max(signal) - np.min(signal)) / (M - 1)

    quantize_signal = delta * np.round(signal / delta)

    quantize_levels = np.arange(np.min(quantize_signal), np.max(quantize_signal) + delta, delta)

    quantize_bit = np.arange(0, M)

    quantize_bit = [format(bits, '0' + str(int(np.log2(M))) + 'b') for bits in quantize_bit]

    quantize_table = np.c_[quantize_levels[:M], quantize_bit[:M]]

    fig, ax = plt.subplots(figsize=(14 / 2.54, M / 2.54))
    table = ax.table(cellText=quantize_table, colLabels=['Значення сигналу', 'Кодова послідовність'], loc='center')
    table.set_fontsize(14)
    table.scale(1, 2)
    ax.axis('off')
    fig.savefig(f'figures/Таблиця квантування для M={M} рівнів.png', dpi=600)

    for signal_value in quantize_signal:
        for index, value in enumerate(quantize_levels[:M]):
            if np.round(np.abs(signal_value - value), 0) == 0:
                bits.append(quantize_bit[index])
                break

    bits = [int(item) for item in list(''.join(bits))]

    fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
    ax.step(np.arange(0, len(bits)), bits, linewidth=0.1)
    ax.set_xlabel('Відліки')
    ax.set_ylabel('Біт')
    ax.set_title(f'Кодова послідовність для M={M} рівнів')
    fig.savefig(f'figures/Кодова послідовність для M={M} рівнів.png', dpi=600)

    dispersion = (delta ** 2) / 12
    snr = (np.var(signal) / dispersion) ** 0.5

    signal_quantized.append(quantize_signal)
    signal_dispersion.append(dispersion)
    signal_snr.append(snr)

fig, axs = plt.subplots(4, 1, figsize=(14 / 2.54, 14 / 2.54), sharex=True)
axs = axs.ravel()
for i, signal in enumerate([signal_quantized[0], signal_quantized[1], signal_quantized[2], signal_quantized[3]]):
    axs[i].step(np.arange(len(signal)), signal, linewidth=0.5)
    axs[i].set_ylabel(f'M={4 * (2 ** i)}')
axs[0].set_title('Цифрові сигнали')
fig.tight_layout()
fig.savefig('figures/Цифрові сигнали.png', dpi=600)

fig, ax = plt.subplots(figsize=(14 / 2.54, 10.5 / 2.54))
ax.plot([4, 16, 64, 256], signal_dispersion, marker='o')
ax.set_xlabel('Кількість рівнів квантування (M)')
ax.set_ylabel('Дисперсія')
ax.set_title('Залежність дисперсії від кількості рівнів квантування')
fig.tight_layout()
fig.savefig('figures/Залежність дисперсії.png', dpi=600)

fig, ax = plt.subplots(figsize=(14 / 2.54, 10.5 / 2.54))
ax.plot([4, 16, 64, 256], signal_snr, marker='o')
ax.set_xlabel('Кількість рівнів квантування (M)')
ax.set_ylabel('Відношення сигнал/шум')
ax.set_title('Залежність відношення сигнал/шум від кількості рівнів квантування')
fig.tight_layout()
fig.savefig('figures/Залежність відношення сигнал-шум.png', dpi=600)