# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal
from matplotlib import pyplot
import scipy.fftpack as fftpack


class GaussWindow(object):

    def __init__(self, size, std = 3):
        self.size = size
        self.std = std
        self.memory = None

    def update(self,data):
        if self.memory is None:
            self.memory = np.asarray(data)
        else:
            self.memory = np.concatenate((self.memory, data), axis=0)

    def next(self):
        if self.memory is not None and self.memory.shape[0] >= self.size:
            # get gauss window 
            window = np.array(scipy.signal.gaussian(self.size, self.std))
            out = np.transpose(np.multiply(window,self.memory.transpose()))

            # slide
            self.memory = self.memory[1:]
            
            return out
        else:
            raise StopIteration()


class IdealFilter (object):
    """ Implements ideal_bandpassing as in EVM_Matlab. """
    
    def __init__(self, wl=.5, wh=.75, fps=1, NFFT=None):
        """Ideal bandpass filter using FFT """

        self.fps = fps
        self.wl = wl
        self.wh = wh
        self.NFFT = NFFT        # windowSize = 40
        
        if self.NFFT is not None:
            self.__set_mask()
            
    def __set_mask(self): 
        self.frequencies = fftpack.fftfreq(self.NFFT, d=1.0/self.fps)    
        # determine what indices in Fourier transform should be set to 0
        self.mask = (np.abs(self.frequencies) < self.wl) | (np.abs(self.frequencies) > self.wh) #频率范围外pixel的index


    def __call__(self, data, axis=0):
        if self.NFFT is None:
            self.NFFT = data.shape[0]
            self.__set_mask()            
        # print("data: ",data,data.shape) 
        fft = fftpack.fft(data, axis=axis)          # data = phases = self.memory
        # print("fft: "+str(fft[fft.nonzero()]))        
        fft[self.mask] = 0   

        ifft_real = np.real(fftpack.ifft(fft, axis=axis))
        # print("IFFT: "+str(ifft_real[ifft_real.nonzero()]))
        return ifft_real
class IdealFilterWindowed_Gaus (GaussWindow):
    
    def __init__(self, winsize, wl=.5, wh=.75, fps=1, step=1, outfun=None):
        GaussWindow.__init__(self,winsize)
        self.filter = IdealFilter(wl, wh, fps=fps, NFFT=winsize)
        # print("selffilter: "+str(self.filter))
        self.outfun = outfun
        
    def next(self):
        out = GaussWindow.next(self)
        out = self.filter(out)
        if self.outfun is not None:
            # apply output function, e.g. to return first (most recent) item
            out = self.outfun(out)

        return out
