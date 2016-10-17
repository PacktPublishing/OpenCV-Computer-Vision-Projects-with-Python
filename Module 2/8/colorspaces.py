import cv2
import numpy as np

def get_frame(cap, scaling_factor):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.5

    while True:
        frame = get_frame(cap, scaling_factor) 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower = np.array([0,70,60])
        upper = np.array([50,150,255])

        # define blue range
        #lower = np.array([60,100,100])
        #upper = np.array([180,255,255])

        # Threshold the HSV image to get only blue color
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        res = cv2.medianBlur(res, 5)

        cv2.imshow('Original image', frame)
        cv2.imshow('Color Detector', res)
        c = cv2.waitKey(5) 
        if c == 27:
            break

    cv2.destroyAllWindows()
