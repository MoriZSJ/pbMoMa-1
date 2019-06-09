from perceptual.filterbank import *
import cv2, sys
import numpy as np
from pyr2arr import Pyramid2arr
from matplotlib import pyplot
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter


def draw_freq(phase,fpsForBandPass,phfftOut,magFreq=False):
    phfft = abs(np.fft.fft(phase))
    freq = np.fft.fftfreq(len(phfft),d = 1./fpsForBandPass)
    # print("phfft: ",phfft)

    i = np.argsort(freq)
    freq = freq[i]
    phfft = phfft[i]
    freq = freq[len(freq)/2+1:]
    phfft = phfft[len(phfft)/2+1:]

    pyplot.figure(figsize=(20, 10))

    pyplot.xlabel("Freq (Hz)")
    pyplot.plot(freq,phfft)
    x_tck = np.arange(min(freq), max(freq)+1,0.2)
    pyplot.xticks(x_tck)
    if magFreq:
        pyplot.title("Magnified FFT")
        pyplot.savefig(magPhfftOut)
    else:
        pyplot.title("FFT")
        pyplot.savefig(phfftOut)
    pyplot.show()



def eyeFreqFilter(vidIn,vidOut,windowSize,factor,lowFreq,highFreq,fpsForBandPass,drawOnce):
    # read input video
    vidReader = cv2.VideoCapture(vidIn)
    vidFrames = int(vidReader.get(cv2.CAP_PROP_FRAME_COUNT))    
    width = int(vidReader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidReader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidReader.get(cv2.CAP_PROP_FPS))
    func_fourcc = cv2.VideoWriter_fourcc

    print("Video: "+str(vidIn), "%d frames" % vidFrames, "(%d x %d)" %(width, height), "FPS:%d" % fps)

    # video Writer

    fourcc = func_fourcc('M', 'J', 'P', 'G')
    vidWriter = cv2.VideoWriter(vidOut, fourcc, fps, (width,height), 1)
    print("Output: "+str(vidOut))

    # initialize the steerable complex pyramid
    steer = Steerable(5)
    pyArr = Pyramid2arr(steer)

    # setup temporal filter
    filter = IdealFilterWindowed(windowSize, lowFreq, highFreq, fps=fpsForBandPass, outfun=lambda x: x[0])

    print("FrameNum: ")
    for FrameNum in range(windowSize+vidFrames):
        print(FrameNum)
        sys.stdout.flush()

        ok,im = vidReader.read()
        if not ok:
            break

        grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        # get coeffs for pyramid
        coeff = steer.buildSCFpyr(grayIm)
        # cv2.imwrite("coeff.png",visualize(coeff))

        # add image pyramid to video array
        arr = pyArr.p2a(coeff)
        phase = np.angle(arr)
        print("phase: ",phase.shape)

        # show phase_freq img
        if FrameNum == 0:
            draw_freq(phase,fpsForBandPass,phfftOut)
        
        # add to temporal filter
        filter.update([phase])

        # try to get filtered output to continue            
        try:
            filteredPhases = filter.next()          # = out
        except StopIteration:
            continue

        print ("done!")

        # motion magnification
        magnifiedPhases = phase + factor*filteredPhases         # one dimension
        # print("magni: ", magnifiedPhases.shape)
        if drawOnce:
            draw_freq(magnifiedPhases,fpsForBandPass,magPhfftOut,magFreq=True)
            drawOnce = False

        # create new array
        newArr = np.abs(arr) * np.exp(magnifiedPhases * 1j)        

        # create pyramid coeffs
        try:
            newCoeff = pyArr.a2p(newArr)
        except StopIteration:
            print("End")

        # reconstruct pyramid
        out = steer.reconSCFpyr(newCoeff)

        # clip values out of range
        out[out>255] = 255
        out[out<0] = 0
        
        # make a RGB image
        rgbIm = np.empty((out.shape[0], out.shape[1], 3))
        rgbIm[:,:,0] = out
        rgbIm[:,:,1] = out
        rgbIm[:,:,2] = out
        
        #write to disk
        res = cv2.convertScaleAbs(rgbIm)
        vidWriter.write(res)

    # free the video reader/writer
    vidReader.release()
    vidWriter.release() 



################# main script
vidIn = "eye_Vid/mybutterfly.mp4"
vidOut = "freq_out/magmybtfy.avi"
phfftOut = "freq_out/phfft.jpg"
magPhfftOut = "freq_out/magphfft.jpg"
drawOnce = True
# the size of the sliding window   #筛选freq的列表长度
windowSize = 40      
# the magnifaction factor
factor = 20
# the fps used for the bandpass (use -1 for input video fps) #筛选freq的范围:[0,fps/2]
fpsForBandPass = 20  
# low ideal filter
lowFreq = 0.01
# high ideal filter
highFreq = 0.1

eyeFreqFilter(vidIn,vidOut,windowSize,factor,lowFreq,highFreq,fpsForBandPass,drawOnce)