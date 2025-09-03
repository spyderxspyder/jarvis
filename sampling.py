import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal

f = 20
t = np.linspace(0,0.5,200)
x1 = np.sin(2*np.pi*f*t)

s_rate = 100 # sampling rate

T = 1/s_rate
n = np.arange(0,0.5/T)
nT = n*T
x2 = np.sin(2 * np.pi * f * nT)

plt.figure(figsize = (10,8))
plt.suptitle('Sampling with Fmax = 20Hz with fs = 35Hz',fontsize = 20)

plt.subplot(2,2,1)
plt.plot(t,x1,linewidth = 3, label = 'SineWave of freq 20Hz')
plt.xlabel('time',fontsize = 15)
plt.ylabel('Amplitude',fontsize = 15)
plt.legend(fontsize = 10,loc = 'upper right')

plt.subplot(2,2,2)
plt.plot(nT,x2,'ro',label = 'Sample marks after resampling at fs = 35Hz')
plt.xlabel('time',fontsize = 15)
plt.ylabel('Amplitude',fontsize = 15)
plt.legend(fontsize = 10,loc = 'upper right')

plt.subplot(2,2,3)
plt.stem(nT,x2,'m',label = 'Sample after resampling at fs = 35Hz')
plt.xlabel('time',fontsize = 15)
plt.ylabel('Amplitude',fontsize = 15)
plt.legend(fontsize = 10,loc = 'upper right')

plt.subplot(2,2,4)
plt.plot(nT,x2,'g-',label = 'Reconstructed Sine Wave')
plt.xlabel('time',fontsize = 15)
plt.ylabel('Amplitude',fontsize = 15)
plt.legend(fontsize = 10,loc = 'upper right')

