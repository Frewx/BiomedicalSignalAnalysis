from biosppy import storage
from biosppy.signals import ecg
import numpy as np
from pylab import * 


import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

from scipy import signal as sg
from scipy.signal import remez
from scipy.signal import freqz
from scipy.signal import lfilter

signal, mdata = storage.load_txt('ecg.txt')
Fs = mdata['sampling_rate']
N = len(signal)  # number of samples
T = (N - 1) / Fs  # duration
ts = np.linspace(0, T, N, endpoint=False)  # relative timestamps
plt.figure(1)
plt.title('ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
#plt.xlim(0,5)
plt.grid()
plt.plot(ts,signal,lw=2)



ftsignal = fft(signal)
ftsignal = abs(ftsignal) ## to get the absolute part of the fft transform
## for normalizing purposes
ftsignal = ftsignal / float(N)
ftsignal = ftsignal**2
##
plt.figure(2)
plt.title('FFT Of The ECG Signal')
plt.ylabel('Power (dB)')
plt.xlabel('Frequency (Hz)')
plt.grid()
plt.plot(20*log10(ftsignal),color = 'G')



highpassfilter = remez(41, [0, 0.1, 0.15, 0.5], [0.0, 1.0]) #highpass 
filtered_signal= lfilter(highpassfilter, 1, signal)

plt.figure(3)
plt.title('Filtered FFT ECG Signal')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.grid()
plt.plot(20*log10(abs(fft((signal)))))
plt.plot(15*log10(abs(fft(filtered_signal))))
plt.legend(('FFTSignal' , 'FilteredFFTSignal'))

w, h = freqz(highpassfilter)
F = w / (2* np.pi)
plt.figure(4)
plt.title(r'Magnitude transfer function in dB')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.plot(F, 20*np.log10(np.abs(h)))


plt.figure(5)
freq, time, spec = sg.spectrogram(signal, Fs)

plt.plot(spec)
#plt.colorbar()
plt.title('STFT Of The ECG Signal')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.xlim(0,20)
#plt.ylim(0,100)

plt.figure(6)
plt.pcolormesh(time,freq,spec)
plt.title('STFT Spectogram Of The ECG Signal')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim(0,100)

plt.show()