import numpy as np
import cv2

img =cv2.resize(cv2.imread("assets/soccer.jpg"),(800,800))
ball =cv2.resize(cv2.imread("assets/ball.png"),(80,80))
h,w=ball.shape[0:2]

methods=[cv2.TM_CCOEFF,cv2.TM_CCOEFF_NORMED,cv2.TM_CCORR,cv2.TM_CCORR_NORMED,
         cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]

for index in methods:
    temp=img.copy()
    #this will return an array after performing convolution on temp 
    # and img with any specific method
    output=cv2.matchTemplate(temp,ball,index)

    #this function will use the above array to find minimum and maximum 
    #point along with their locations
    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(output)

    if index in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        #this is done because for Square difference & Normalized Square Difference 
        # method min value represents the best fit
        start_point=min_loc
    else:
        #whereas, for rest of the methods maximum value represents the 
        #best fit
        start_point=max_loc

    #Calculating the ending point
    ending_point=(start_point[0]+w,start_point[1]+h)
    cv2.rectangle(temp,start_point,ending_point,255,2)

    cv2.imshow("img",temp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    