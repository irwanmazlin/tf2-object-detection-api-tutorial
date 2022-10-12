import cv2 as cv
import os
import argparse

def vidToFrame(input, output):
    count = 0 
    cap = cv.VideoCapture(input)
    success, img = cap.read()

    while success:
        cap.set(cv.CAP_PROP_POS_MSEC, (count*1000))
        success, img = cap.read()
        print("new frames : ", success)
        cv.imwrite(output + "/frame%d.jpg" % count, img)
        count = count + 1
    

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--input", help="video directory")
    a.add_argument("--output", help="frame directory")
    args = a.parse_args()
    print(args)
    vidToFrame(args.input, args.output)
