import cv2
import numpy as np


def load_classifier():
    return cv2.CascadeClassifier("./haarcascade_eye.xml")


def detect_eyes(cam_img):
    gray = cam_img
    faces = load_classifier()

    detected = faces.detectMultiScale(gray, 1.3, 5)

    pupilFrame = gray
    pupilO = gray
    windowClose = np.ones((5, 5), np.uint8)
    windowOpen = np.ones((2, 2), np.uint8)
    windowErode = np.ones((2, 2), np.uint8)

    # draw square
    for (x, y, w, h) in detected:
        cv2.rectangle(gray, (x, y), ((x + w), (y + h)), (0, 0, 255), 1)
        cv2.line(gray, (x, y), ((x + w, y + h)), (0, 0, 255), 1)
        cv2.line(gray, (x + w, y), ((x, y + h)), (0, 0, 255), 1)
        pupilFrame = cv2.equalizeHist(gray[y + int((h * .25)):int((y + h)), x:int((x + w))])
        pupilO = pupilFrame
        ret, pupilFrame = cv2.threshold(pupilFrame, 55, 255, cv2.THRESH_BINARY)  # 50 ..nothin 70 is better
        pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_CLOSE, windowClose)
        pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_ERODE, windowErode)
        pupilFrame = cv2.morphologyEx(pupilFrame, cv2.MORPH_OPEN, windowOpen)

        # so above we do image processing to get the pupil..
        # now we find the biggest blob and get the centriod


        threshold = cv2.inRange(pupilFrame, 250, 255)  # get the blobs
        _, contours, hierarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # if there are 3 or more blobs, delete the biggest and delete the left most for the right eye
        # if there are 2 blob, take the second largest
        # if there are 1 or less blobs, do nothing

        if len(contours) >= 2:
            # find biggest blob
            maxArea = 0
            MAindex = 0  # to get the unwanted frame
            distanceX = []  # delete the left most (for right eye)
            currentIndex = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                center = cv2.moments(cnt)
                if center['m00'] != 0:
                    cx, cy = int(center['m10'] / center['m00']), int(center['m01'] / center['m00'])
                    distanceX.append(cx)
                    if area > maxArea:
                        maxArea = area
                        MAindex = currentIndex
                    currentIndex = currentIndex + 1

            del contours[MAindex]  # remove the picture frame contour
            del distanceX[MAindex]

        eye = 'right'

        if len(contours) >= 2:  # delete the left most blob for right eye
            if eye == 'right':
                edgeOfEye = distanceX.index(min(distanceX))
            else:
                edgeOfEye = distanceX.index(max(distanceX))
            del contours[edgeOfEye]
            del distanceX[edgeOfEye]

        if len(contours) >= 1:  # get largest blob
            maxArea = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > maxArea:
                    maxArea = area
                    largeBlob = cnt

        if len(largeBlob) > 0:
            center = cv2.moments(largeBlob)
            cx, cy = int(center['m10'] / center['m00']), int(center['m01'] / center['m00'])
            cv2.circle(pupilO, (cx, cy), 5, 255, -1)
        cv2.imshow('PUPIL', pupilO)

        cv2.imshow('PUPIL FRAME', pupilFrame)
    return (gray)


def start_video(cam_num):
    cap = cv2.VideoCapture(cam_num)

    while True:
        # grab frames

        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # process pupils
        gray = detect_eyes(gray)
        # continious image render from camera
        cv2.imshow("TEST WINDOW", gray)

        # wait for keyboard interrupt
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_video(0)
