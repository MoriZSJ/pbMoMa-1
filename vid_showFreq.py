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
def show_frequencies(vid_data,freqOut,framerate,bounds=None):
    """Graph the frequency strength of the video"""
    averages = []
    if bounds:
        for x in range(0, vid_data.shape[0]):
            averages.append(vid_data[x, bounds[2]:bounds[3], bounds[0]:bounds[1], :].sum())
    else:
        for x in range(1, vid_data.shape[0]-1):
            averages.append(vid_data[x, :, :, :].sum()) # each channel has equal greyvalue

    averages = averages - min(averages)
    # print("aver: ",averages.shape)

    freqs = scipy.fftpack.fftfreq(len(averages), d=1.0 / framerate)  #生成采样频率(signal size, sampling interval)
    # print("freq: ",freqs.shape, freqs)
    fft = abs(scipy.fftpack.fft(averages))
    freqs = freqs[1:len(freqs) / 2 ]
    fft = fft[1:len(fft) / 2 ]
    # print("freqs: ",len(freqs), "\nfft: ",len(fft))

    pyplot.figure(figsize=(20, 10))
    pyplot.xlabel("Freq (Hz)")
    pyplot.ylabel("|Average(Freq)|")
    pyplot.plot(freqs, fft/sum(fft))
    pyplot.title("FFT")
    # print("fft shape: ",abs(fft).shape,abs(fft)[:20])
    x_tck = np.arange(min(freqs),max(freqs)+0.01,(max(freqs)-min(freqs))/40)
    pyplot.xticks(x_tck)
    pyplot.savefig(freqOut)
    pyplot.show()
#######



########### main script
vidFname = "mag_Videos/btfy/test4_newphase.avi"
freqOut = "mag_Videos/btfy/vidfft4_np.jpg"
framerate = 30
vid = vid_read(vidFname)
# em.show_frequencies(vid, framerate)
show_frequencies(vid, freqOut, framerate) #,bounds=[0,180,0,320])
