# OpenCV Python program to detect moving objects in frame
# import libraries of python OpenCV
import cv2
import cv
import numpy as np
import Queue

# capture frames from a video
cap = cv2.VideoCapture("myVideos/custom1.mp4")

# setup output file
codec = cv.CV_FOURCC('D','I','V','X')
videoFile = cv2.VideoWriter();
videoFile.open('output/1custom.avi', codec, 25, (480, 270),1)


# setup background subtractor object
# car_cascade = cv2.CascadeClassifier('data/cars.xml')
bgsMOG = cv2.BackgroundSubtractorMOG()

# setup queue for tracking objects
q = Queue.Queue(maxsize=50) # maxsize is how long to keep direction vectors

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        # make image grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # apply gaussian blur to improve detection
        gray = cv2.GaussianBlur(gray, (31, 31), 0)

        # apply backgorund subtractor mask
        fgmask = bgsMOG.apply(gray, None, 0.01)
        # cv2.imshow('Mask', fgmask)

        # To find the contours of the objects
        contours, hierarchy = cv2.findContours(fgmask,
                                cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        for contour, hier in zip(contours, hierarchy):

            # draw bounding rectangle
            (x, y, w, h) = cv2.boundingRect(contour)

            # only show if the image is big enough
            if w > 5 and h > 5:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 255), 2)

                # pop from queue if it is full
                if (q.full()):
                    q.get()

                # add location to queue
                q.put((x+w/2, y+h/2))

                # iterate through queue and draw green rectangles on frame for tracking
                for elem in list(q.queue):
                    cv2.rectangle(frame, (elem), (elem[0] + 4, elem[1] + 2), (0, 255, 0), -1);

        # show frame live
        cv2.imshow('Output', frame)

        # save frame to file
        videoFile.write(frame)

        key = cv2.waitKey(60)
        if key == 27:
            break

# clear video capture
cap.release()

# clear memory
cv2.destroyAllWindows()
