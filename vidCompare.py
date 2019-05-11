import cv2
import sys
import numpy as np

def vidCom(vid1, vid2,vidFnameOut):
    print ("Reading:", vid1)
    # get vid properties
    vidRead1 = cv2.VideoCapture(vid1)
    vidFrames = int(vidRead1.get(cv2.CAP_PROP_FRAME_COUNT))    
    width = int(vidRead1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidRead1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vidRead1.get(cv2.CAP_PROP_FPS))
    func_fourcc = cv2.VideoWriter_fourcc

    print ("Reading:", vid2)
    vidRead2 = cv2.VideoCapture(vid2)
    vidFrames2 = int(vidRead2.get(cv2.CAP_PROP_FRAME_COUNT))    
    width2 = int(vidRead2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(vidRead2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = int(vidRead2.get(cv2.CAP_PROP_FPS))
    func_fourcc2 = cv2.VideoWriter_fourcc

    # input video should have same properties
    assert vidFrames==vidFrames2
    assert width==width2
    assert height==height2
    assert fps==fps2
    assert func_fourcc==func_fourcc2

    # video Writer
    rst = np.zeros((vidFrames,height,width,3))
    fourcc = func_fourcc('M', 'J', 'P', 'G')
    vidWriter = cv2.VideoWriter(vidFnameOut, fourcc, int(fps), (width,height), 1)
    print ('Writing:', vidFnameOut)

    for frame in range(vidFrames):
        print("frame: "+str(frame))
        sys.stdout.flush()

        _,im1 = vidRead1.read()
        _,im2 = vidRead2.read()

        rst[frame] = np.abs(im1-im2)  #640x360
        for row in range(rst[frame].shape[0]):
            #print(row,rst[frame].shape[0])
            for col in range(rst[frame].shape[1]):
                #print("pix: "+str(pix)+str(pix.shape))
                if (rst[frame,row,col]<90).all():
                    #print(str(rst[frame,row,col]))
                    rst[frame,row,col] = [0,0,0]
                    #print(str(rst[frame,row,col]))
        res = cv2.convertScaleAbs(rst[frame]/10)
        vidWriter.write(res)
        #print("im1: "+str(im1.ndim)+'\nim2: '+str(im2.ndim)) # ndim=3
        #print(im1,im2)
        # if (np.abs(im1-im2)).all()<2:
        #     vidWriter.write(0*np.abs(im1-im2))
        # else:
        #     vidWriter.write(np.abs(im1-im2))
       
    print("Done")
    print("RST: "+str(rst))
    vidRead1.release()
    vidRead2.release()
    vidWriter.release()



#================= main script ==================
vid1 = 'eye/eye-ud.mp4-origin.avi'
vid2 = 'eye/eye-ud.mp4-Mag300Ideal-lo0.10-hi1.00-fps100.avi'

vidFnameOut = 'eye/vidComp-'+vid2[-24:-4]+'.avi'

vidCom(vid1, vid2,vidFnameOut)