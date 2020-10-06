import cv2
import numpy as np
import time

cap=cv2.VideoCapture(0)
time.sleep(3)
backgnd=0
count=0

for x in range(30):
    ret, backgnd=cap.read()
    if not ret:
        continue
backgnd=np.flip(backgnd,axis=1)

"""
Color Detection and reading from video
"""
while (cap.isOpened()):
    # capturing the live frame
    ret_val,img=cap.read()
    if not ret_val:
        break
    count=count+1
    # flipping the image
    img=np.flip(img, axis=1)

    # converting BGR to HSV space
    hsv= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    #Range for lower range Red
    lower_red=np.array([0,120,70])
    upper_red=np.array([10,255,255])
    mask1=cv2.inRange(hsv,lower_red,upper_red)

    #Range for upper range Red
    lower_red=np.array([170,120,70])
    upper_red=np.array([180,255,255])
    mask2=cv2.inRange(hsv,lower_red,upper_red)

    #Generating final mask for red color
    mask1=mask1+mask2

    """
    Segmenting out the red color
    """

    mask1=cv2.morphologyEx(mask1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=2)
    mask1=cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations=1)

    #Creating an inverted mask for segmenting
    mask2=cv2.bitwise_not(mask1)

    #Segmenting the cloth out of the frame using bitwise and with the inverted mask
    res1=cv2.bitwise_and(img,img, mask=mask2)

    """
    Generating the final output
    """

    res2=cv2.bitwise_and(backgnd,backgnd,mask=mask1)
    #Generating the final output
    output=cv2.addWeighted(res1,1,res2,1,0)
    cv2.imshow("Now YOu cant see me",output)

    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
