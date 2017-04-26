import cv2
import numpy as np

bgsMOG = cv2.BackgroundSubtractorMOG()
cap    = cv2.VideoCapture("videos/streetSmall.mp4")
counter = 0

if cap:
    while True:
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = bgsMOG.apply(gray, None, 0.01)
            cv2.imshow('Mask', fgmask)
            # cv2.line(frame,(0,60),(160,60),(255,255,0),1)
            # To find the countours of the Cars
            contours, hierarchy = cv2.findContours(fgmask,
                                    cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            try:
                hierarchy = hierarchy[0]

            except:
                hierarchy = []

            for contour, hier in zip(contours, hierarchy):
                (x, y, w, h) = cv2.boundingRect(contour)

                if w > 5 and h > 5:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 255), 1)

                    #To find centroid of the Car
#                     x1 = w/2
#                     y1 = h/2
#
#                     cx = x+x1
#                     cy = y+y1
# ##                    print "cy=", cy
# ##                    print "cx=", cx
#                     centroid = (cx,cy)
##                    print "centoid=", centroid
                    # Draw the circle of Centroid
#                     cv2.circle(frame,(int(cx),int(cy)),2,(0,0,255),-1)
#
#                     # To make sure the Car crosses the line
# ##                    dy = cy-108
# ##                    print "dy", dy
#                     if centroid > (27, 38) and centroid < (134, 108):
# ##                        if (cx <= 132)and(cx >= 20):
#                         counter +=1
##                            print "counter=", counter
##                    if cy > 10 and cy < 160:
                    # cv2.putText(frame, str(counter), (x,y-5),
                    #                     cv2.FONT_HERSHEY_SIMPLEX,
                    #                     0.5, (255, 0, 255), 2)
##            cv2.namedWindow('Output',cv2.cv.CV_WINDOW_NORMAL)
            cv2.imshow('Output', frame)
##          cv2.imshow('FGMASK', fgmask)


            key = cv2.waitKey(60)
            if key == 27:
                break

cap.release()
cv2.destroyAllWindows()
