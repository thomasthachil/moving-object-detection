# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV
import numpy as np
import cv2
import cv

# capture frames from a video
cap = cv2.VideoCapture('videos/naveSmall.mp4')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('data/cars.xml')
ret, frame = cap.read()

# Set up output file
# codec = cv.CV_FOURCC('D','I','V','X')
# videoFile = cv2.VideoWriter();
# videoFile.open('output/cascadeDetection.avi', codec, 25, (480, 270),1)

# loop runs if capturing has been initialized.
while(cap.isOpened()):
    # reads frames from a video
    ret, frame = cap.read()

    # cv2.imshow('Vehicle Detection Original', frame)
    # convert to gray scale for each each frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects cars of different sizes in the input image using classifier
    cars = car_cascade.detectMultiScale(gray, 1.5, 2)

    # To draw a rectangle for each car
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1, 4)

        # show frames live
        cv2.imshow('Vehicle Detection', frame)

        # save frames to output file
        # videoFile.write(frame)

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

# clear capture
cap.release()

# clear memory
cv2.destroyAllWindows()
