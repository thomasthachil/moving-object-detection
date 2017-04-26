
# OpenCV Python program to detect cars in video frame
# import libraries of python OpenCV
import numpy as np
import cv2

# capture frames from a video
# bgsMOG = cv2.BackgroundSubtractorMOG()
cap = cv2.VideoCapture('videos/trafficSmall.mp4')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('data/cars.xml')
# hog = cv2.HOGDescriptor("data/vehicle_detector.yml")
ret, frame = cap.read()

# loop runs if capturing has been initialized.
while(cap.isOpened()):
    # reads frames from a video
    ret, frame = cap.read()
    # cv2.imshow('Vehicle Detection Original', frame)
    # convert to gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # fgmask = bgsMOG.apply(frame)
    # cv2.imshow('Vehicle Detection', fgmask)
    # Detects cars of different sizes in the input image
    cars = car_cascade.detectMultiScale(gray, 1.5, 2)
    # To draw a rectangle in each cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),1, 4)

        # Display frames in a window
        cv2.imshow('Vehicle Detection', frame)

    # Wait for Esc key to stop
    if cv2.waitKey(33) == 27:
        break

cap.release;
# De-allocate any associated memory usage
cv2.destroyAllWindows()
