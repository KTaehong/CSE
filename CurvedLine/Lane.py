# import packages
import numpy as np
import cv2
import time


cap=cv2.VideoCapture('road.mov')
arrow = cv2.imread('arrow.png')
while(True):
    #Make a copy of original image
    ret,frame = cap.read()
    if frame is not None:
        # I want to put logo on top-left corner, So I create a ROI
        rows,cols,channels = arrow.shape
        roi = frame[0:rows, 0:cols]
        # Now create a mask of logo and create its inverse mask also
        arrowgray = cv2.cvtColor(arrow,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(arrowgray,215, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(roi,roi,mask = mask)
        arrow_fg = cv2.bitwise_and(arrow,arrow,mask = mask_inv)
        dst = cv2.add(frame_bg,arrow_fg)
        frame[0:rows, 0:cols ] = dst
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray,215,255,cv2.THRESH_BINARY)
        cv2.imshow('added',frame)
        cv2.imshow('gray',thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

