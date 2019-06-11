from perceptual.filterbank import *
import cv2, sys
import numpy as np
from pyr2arr import Pyramid2arr
from matplotlib import pyplot
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter


def draw_freq(phase,fpsForBandPass,phfftOut,magFreq=False):
    phfft = abs(np.fft.fft(phase))
    freq = np.fft.fftfreq(len(phfft),d = 1./fpsForBandPass)
 

    freq = freq[1:len(freq)/2]
    phfft = phfft[1:len(phfft)/2]
    # print("freq: ",min(freq))

    pyplot.figure(figsize=(20, 10))

    pyplot.xlabel("Freq (Hz)")
    pyplot.plot(freq,phfft)
    x_tck = np.arange(min(freq), max(freq),(max(freq)-min(freq))/20)
    pyplot.xticks(x_tck)
    # y_tck = np.arange(min(phfft), 100000,(100000-min(phfft))/10)
    # pyplot.yticks(y_tck)
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
        magnifiedPhases = factor*filteredPhases         # one dimension
        # print("magni: ", magnifiedPhases.shape)
        if drawOnce:
            draw_freq(filteredPhases,fpsForBandPass,magPhfftOut,magFreq=True)
            drawOnce = False

        # create new array
        newArr = np.abs(arr) * np.exp(magnifiedPhases * 1j)        

        # create pyramid coeffs
        try:
            newCoeff = np.asarray(pyArr.a2p(newArr))
            # print("coeff1: ",newCoeff[3])
            for i in range(len(newCoeff)):
                if i <= 3:
                    newCoeff[i] = np.array(newCoeff[i]) - np.array(newCoeff[i])  
                else:
                    continue                    

            # print("coeff111: ",newCoeff[3])
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
        
        # write to disk
        res = cv2.convertScaleAbs(rgbIm)
        vidWriter.write(res)

    # free the video reader/writer
    vidReader.release()
    vidWriter.release() 



################# main script
vidIn = "eye_Vid/eye-btfy.mp4"
vidOut = "mag_Videos/btfy/testtt.avi"
phfftOut = "mag_Videos/btfy/phfft.jpg"
magPhfftOut = "mag_Videos/btfy/phfft_mag.jpg"
drawOnce = True
# the size of the sliding window   #筛选freq的列表长度
windowSize = 50    
# the magnifaction factor
factor = -1
# the fps used for the bandpass (use -1 for input video fps) #筛选freq的范围:[0,fps/2]
fpsForBandPass = 30
# low ideal filter
lowFreq = 0.7
# high ideal filter
highFreq = 10

eyeFreqFilter(vidIn,vidOut,windowSize,factor,lowFreq,highFreq,fpsForBandPass,drawOnce)