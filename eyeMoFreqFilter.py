from perceptual.filterbank import *
import cv2, sys
import numpy as np
from pyr2arr import Pyramid2arr
from temporal_filters import IdealFilterWindowed, ButterBandpassFilter

def eyeFreqFilter(vidIn,vidOut,windowSize,lowFreq,highFreq,fpsForBandPass):
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

        _,im = vidReader.read()
        grayIm = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        # get coeffs for pyramid
        coeff = steer.buildSCFpyr(grayIm)
        cv2.imwrite("coeff.png",visualize(coeff))
        # add image pyramid to video array
        arr = pyArr.p2a(coeff)
        print("arr: ",arr)



    # free the video reader/writer
    vidReader.release()
    vidWriter.release() 



################# main script

vidIn = 'eye_Vid/eye-ud.mp4' 
vidOut = 'freq_out/test_freq.avi'
  
# the size of the sliding window   #筛选freq的列表长度
windowSize = 40      
# the fps used for the bandpass (use -1 for input video fps) #筛选freq的范围:[0,fps/2]
fpsForBandPass = 0.3   
# low ideal filter
lowFreq = 0.01
# high ideal filter
highFreq = 0.1

eyeFreqFilter(vidIn,vidOut,windowSize,lowFreq,highFreq,fpsForBandPass)