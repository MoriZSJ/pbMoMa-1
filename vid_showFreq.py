import eulerian_magnification as em
import scipy.fftpack
from matplotlib import pyplot
import numpy as np
import cv2
import sys

######## save video as array
def vid_read(vidFname):
    vidRead = cv2.VideoCapture(vidFname)
    vidFrames = int(vidRead.get(cv2.CAP_PROP_FRAME_COUNT)) 
    print("frames: ",vidFrames)   
    width = int(vidRead.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidRead.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidRead.get(cv2.CAP_PROP_FPS))

    vid = np.zeros((vidFrames,height,width,3))
    print(height,width)
    for frame in range(vidFrames):
        _, vid[frame] = vidRead.read()
    return vid
######### 

########### show frame
# cv2.imshow("frame", vid[40])
# cv2.waitKey(0)
#############

####### show freq
def show_frequencies(vid_data,framerate,bounds=None):
    """Graph the frequency strength of the video"""
    averages = []
    if bounds:
        for x in range(0, vid_data.shape[0]):
            averages.append(vid_data[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
    else:
        for x in range(1, vid_data.shape[0]-1):
            averages.append(vid_data[x, :, :, :].sum()) # each channel has equal greyvalue

    averages = averages - min(averages)
    print("aver: ",averages)

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / framerate)  #生成采样频率(signal size, sampling freq)
    # print("fft: ",scipy.fftpack.fft(averages))
    fft = abs(scipy.fftpack.fft(averages))
    # print("fft: ",fft)
    idx = np.argsort(freqs)     # sort the list index
    pyplot.plot()                # pyplot.subplot(charts_y, charts_x, 2)
    pyplot.title("FFT")
    pyplot.xlabel("Freq (Hz)")
    freqs = freqs[idx]
    # print("freqs: ",freqs)
    fft = fft[idx]  
    # print("fft: ",fft)

    freqs = freqs[len(freqs) / 2 +1:]
    fft = fft[len(fft) / 2 +1:]
    # fft = np.where(fft>0,fft,0)
    print("freqs: ",freqs,"\nfft: ",fft)
    pyplot.plot(freqs, fft/sum(abs(fft)))
    # print("fft shape: ",abs(fft).shape,abs(fft)[:20])
    x_tck = np.arange(min(freqs),max(freqs),10)
    pyplot.xticks(x_tck)
    pyplot.show()
#######



########### main script
vidFname = "test/guitar.mp4"
framerate = 600
vid = vid_read(vidFname)
# em.show_frequencies(vid, framerate)
show_frequencies(vid, framerate) #,bounds=[0,180,0,320])
