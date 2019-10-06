import cv2, time, os
import pandas as pd
from datetime import datetime

FILE_OUTPUT = 'sec_video.avi'

class MotionDetection:
    def __init__(self):
        #Capture video from camera.
        self.video = cv2. VideoCapture(0)

        self.static_back = None # Initialize comparing frame.
        self.pre_motion = False # Initialize previous motion status.

        # Check if video exists and delet it.
        if os.path.isfile(FILE_OUTPUT):
            os.remove(FILE_OUTPUT)

        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)) # Video width.
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Video heidht.

        # Create de video record.
        self.output = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (width,height))

    def main(self):
        while (self.video.isOpened()):
            motion = False
            check, frame = self.video.read()    # Get frames from video.
            if not check:
                break

            # Check moves and record them.
            if self.pre_motion:
                self.output.write(frame)

            # Convert frame to black&white and blur it.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21,21), 0)

            # Check the first frame.
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

            # Go through every contour.
            for c in contours:
                # Check the size of the contour
                if cv2.contourArea(c) < 5000:
                    continue

                motion = True
                # Create a rectangle with move dimension
                (x,y,w,h) = cv2.boundingRect(c)
                # Draw a green rectangle in the moving object.
                cv2.rectangle(frame, (x,y), (x+w, y+w), (0, 255,0), 2)


            # Show the image.
            cv2.imshow('frame', frame)

            # Check moves.
            if self.pre_motion != motion:
                if self.pre_motion == False:
                    print("Move detected: ", datetime.now())
                else:
                    print("Move finished: ", datetime.now())
                self.pre_motion = motion

            # Check for 'Ctrl+c' command to finish
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video.release()
        self.output.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    md = MotionDetection()
    md.main()
