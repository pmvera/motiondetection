import cv2, time
import pandas as pd
from datetime import datetime


class MotionDetection:
    def __init__(self):
        self.video = cv2. VideoCapture(0)    #Capture video from camera.
        self.static_back = None
        self.pre_motion = False


    def main(self):
        while (True):
            check, frame = self.video.read()   #get frames from video.
            if not check:
                break

            motion = False

            # Convert frame to black&white and blur it.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21,21), 0)

            if self.static_back is None:
                self.static_back = gray
                continue

            # Get the diference between background and current frame.
            diff = cv2.absdiff(self.static_back, gray)

            # If the diference between current frame and backgorund is greater than '25' show white color
            thres = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thres = cv2.dilate(thres, None, iterations=2)

            # Find contours with the threshold.
            im, contours, hierarchy = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < 5000:
                    continue

                motion = True
                (x,y,w,h) = cv2.boundingRect(c)
                # Draw a green rectangle in the moving object.
                cv2.rectangle(frame, (x,y), (x+w, y+w), (0, 255,0), 2)

            # Show the image.
            cv2.imshow('frame', frame)
            if self.pre_motion != motion:
                if self.pre_motion == False:
                    print("Move detected: ", datetime.now())
                else:
                    print("Move finished: ", datetime.now())
                self.pre_motion = motion

            # Check for 'Ctrl+c' command to finish
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    md = MotionDetection()
    md.main()
