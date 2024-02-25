import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt

n =500
Fs = 1000
random = np.random.normal(0, 10, n)

time = np.arange(n)/Fs

F_max =9
w = F_max/(Fs/2)
parameters_filter = signal.butter(3, w, 'low', output='sos')

filtered_signal = signal.sosfiltfilt(parameters_filter, random)

def plot_signal(y, x, title, xlabel, ylabel):
  fig, ax = plt.subplots(figsize=(21/2.54, 14/2.54))
  ax.plot(x, y, linewidth=1)
  ax.set_xlabel(xlabel, fontsize=14)
  ax.set_ylabel(ylabel, fontsize=14)
  plt.title(title, fontsize=14)
  fig.savefig('./figures/'+title+'.png', dpi=600)

plot_signal(filtered_signal, time, 'Filtered signal', 'Time, s', 'Amplitude')

spectrum = fft.fft(filtered_signal)
spectrum = np.abs(fft.fftshift(spectrum))
freq = fft.fftfreq(n, 1/n)
freq = fft.fftshift(freq)

plot_signal(spectrum, freq, 'Spectrum', 'Frequency, Hz', 'Spectrum, dB')