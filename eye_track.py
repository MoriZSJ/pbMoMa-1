import cv2
import sys
import numpy as np

def eyeTrack(video,center):

    #check OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    print("cv: "+str(cv2.__version__))
    # if int(minor_ver) < 3:
    #     tracker = cv2.Tracker_create(tracker_type)
    # else:
    #     if tracker_type == 'BOOSTING':
    #         tracker = cv2.TrackerBoosting_create()
    #     if tracker_type == 'MIL':
    #         tracker = cv2.TrackerMIL_create()
    #     if tracker_type == 'KCF':
    #         tracker = cv2.TrackerKCF_create()
    #     if tracker_type == 'TLD':
    #         tracker = cv2.TrackerTLD_create()
    #     if tracker_type == 'MEDIANFLOW':
    #         tracker = cv2.TrackerMedianFlow_create()

    # Choose tracker
    # tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE','CSRT']
    # tracker_type = tracker_types[2]
    tracker = cv2.TrackerKCF_create()   # KCF Tracker
    #tracker = cv2.TrackerGOTURN_create() # GOTURN Tracker
    # Exit if video not opened.
    if not video.isOpened():
        print('Could not open video') 
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    #bbox = (287, 23, 86, 320)

    # Manually select a bounding box
    bbox = cv2.selectROI(frame, False) #get ordinates of bbox, i.e (89.0, 245.0, 177.0, 69.0)
    #print("bbox: "+str(bbox))
    # get box-center
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    center.append((int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2)))
    #print("centers: "+str(center))
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    if not ok:
        print('Track Failed')

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            cv2.waitKey()
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success, draw bbox
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2)
            
            #add new center
            center.append((int((p1[0]+p2[0])/2),int((p1[1]+p2[1])/2)))
            
        else :
            # Tracking failure
            cv2.putText(frame, 'Tracking failure detected', (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, ' Tracker', (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)

        # Display FPS on frame
        cv2.putText(frame, 'FPS : ' + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv2.imshow('Tracking', frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    
    return center


def drawPath(center):
    #print("center: "+str(center))
    # white background
    rows, cols = [1100,700]
    img = np.zeros((cols,rows,3), np.uint8)
    img.fill(255)

    #draw path
    for i in range(len(center)-1):
        cv2.line(img,center[i],center[i+1],(0,0,0),3)

    #translate
    #M = np.float32([[1,0,-50],[0,1,-150]])
    #img = cv2.warpAffine(img,M,(cols,rows))

    cv2.imshow('PathOri', img)
    cv2.waitKey()
    cv2.imwrite('Result/btf-1064-Ampfps10.jpg', img)
    
    # Image Amplification
    # img2 = cv2.imread('ori.jpg')
    # imgscl=cv2.resize(img2,(1000,1000),interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('PathScaled', imgscl)
    # cv2.imwrite('Scaled.jpg', imgscl)e
    # cv2.waitKey()

#def imgProcess():



if __name__ == '__main__' :
    
    # Read video
    video = cv2.VideoCapture('eye/test.avi')
        
    #center locations
    center = []

    eyeTrack(video,center)
    drawPath(center)
    #imgProcess()