import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

n = 500
Fs = 1000
F_max = 9
F_filter = 16
t = np.linspace(0, n / Fs, n, endpoint=False)
input_signal = np.random.randn(n)

w = F_filter / (Fs / 2)
b, a = signal.butter(3, w, 'low', output='ba')
filtered_signal = signal.filtfilt(b, a, input_signal)

discrete_signals = []
discrete_spectrums = []
reconstructed_signals = []
dispersions = []
snrs = []

Dts = [2, 4, 8, 16]
for Dt in Dts:
    discrete_signal = np.zeros(n)
    for i in range(0, round(n / Dt)):
        discrete_signal[i * Dt] = filtered_signal[i * Dt]
    discrete_signals.append(discrete_signal)


    fft_discrete = np.fft.fft(discrete_signal)
    fft_discrete = np.fft.fftshift(fft_discrete)
    freq = np.fft.fftfreq(n, 1 / Fs)
    freq = np.fft.fftshift(freq)
    discrete_spectrums.append(abs(fft_discrete))


    w = F_filter / (Fs / 2)
    b, a = signal.butter(3, w, 'low', output='ba')
    reconstructed_signal = signal.filtfilt(b, a, discrete_signal)
    reconstructed_signals.append(reconstructed_signal)

    E1 = reconstructed_signal - filtered_signal
    signal_var = np.var(filtered_signal)
    E1_var = np.var(E1)
    dispersions.append(E1_var)
    snrs.append(signal_var / E1_var)

fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
line_width = 1
font_size = 14
x = t
s = 0
for i in range(2):
   for j in range(2):
       ax[i, j].plot(x, discrete_signals[s], linewidth=line_width)
       s += 1
fig.supxlabel('Час, с', fontsize=font_size)
fig.supylabel('Амплітуда', fontsize=font_size)
fig.suptitle('Дискретизовані сигнали', fontsize=font_size)
fig.savefig('./figures/discretized_signals.png', dpi=600)

fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
line_width = 1
font_size = 14
x = freq
s = 0
for i in range(2):
   for j in range(2):
       ax[i, j].plot(x, discrete_spectrums[s], linewidth=line_width)
       s += 1
fig.supxlabel('Частота, Гц', fontsize=font_size)
fig.supylabel('Амплітуда', fontsize=font_size)
fig.suptitle('Спектри дискретизовані сигналів', fontsize=font_size)
fig.savefig('./figures/discrete_spectrums.png', dpi=600)

fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
line_width = 1
font_size = 14
x = t
s = 0
for i in range(2):
   for j in range(2):
       ax[i, j].plot(x, reconstructed_signals[s], linewidth=line_width)
       s += 1
fig.supxlabel('Час, с', fontsize=font_size)
fig.supylabel('Амплітуда', fontsize=font_size)
fig.suptitle('Відновленні  сигнали', fontsize=font_size)
fig.savefig('./figures/reconstructed_signals.png', dpi=600)

fig, ax = plt.subplots(figsize=(14 / 2.54, 10 / 2.54))
line_width = 1
font_size = 14
x = Dts
y = dispersions
ax.plot(x, y, linewidth=line_width)
ax.set_xlabel('Крок дискретизація', fontsize=font_size)
ax.set_ylabel('Дисперсія', fontsize=font_size)
ax.set_title('Залежність дисперсії від кроку дискретизації', fontsize=font_size)
fig.savefig('./figures/dispersion.png', dpi=600)

fig, ax = plt.subplots(figsize=(14 / 2.54, 10 / 2.54))
line_width = 1
font_size = 14
x = Dts
y = snrs
ax.plot(x, y, linewidth=line_width)
ax.set_xlabel('Крок дискретизації', fontsize=font_size)
ax.set_ylabel('Відношення сигнал/шум', fontsize=font_size)
ax.set_title('Залежність відношення сигнал/шум від кроку дискретизації', fontsize=font_size)
fig.savefig('./figures/snr.png', dpi=600)