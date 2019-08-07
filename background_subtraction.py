from __future__ import print_function
import cv2 as cv

vid_file = 'videos/16.avi'

#create Background Subtractor objects
backSub = cv.createBackgroundSubtractorMOG2(varThreshold=15, detectShadows=False)

# [capture]
capture = cv.VideoCapture(vid_file)
if not capture.isOpened:
    print('Unable to open: ' + vid_file)
    exit(0)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc('M','P','E','G')

# Writes the video as output16.avi
out = cv.VideoWriter('output16.avi',fourcc, 1.0, (1148,862), isColor = False)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    ## [apply]
    #update the background model
    fgMask = backSub.apply(frame, None, -1)

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

    out.write(fgMask)
    ## [show]
    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


# added test comments
