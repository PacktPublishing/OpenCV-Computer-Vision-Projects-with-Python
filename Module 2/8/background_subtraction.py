import cv2
import numpy as np

def get_frame(cap, scaling_factor):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    bgSubtractor = cv2.BackgroundSubtractorMOG()
    history = 100

    while True:
        frame = get_frame(cap, 0.5)
        mask = bgSubtractor.apply(frame, learningRate=1.0/history)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Input frame', frame)
        cv2.imshow('Moving Objects', mask & frame)
        c = cv2.waitKey(10)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
