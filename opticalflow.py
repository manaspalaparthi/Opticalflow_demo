import numpy as np
import cv2
import time
import threading, time
from camera import WebcamStream

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    img_bgr = cv2.resize(img_bgr, (0, 0), fx=2, fy=2)

    return img_bgr


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


# cap = cv2.VideoCapture("AngleSampleData_Site3_KaroondaPark_080622_50m_5ms_90degrees.MP4")
# suc, prev = cap.read()
# prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

webcam_stream = WebcamStream(stream_id="sample/Bad Way to Wake Up __ ViralHog.mp4") # 0 id for main camera
#webcam_stream = WebcamStream(stream_id=0) # 0 id for main camera
webcam_stream.start()

prev = webcam_stream.read()
prevgray = cv2.resize(prev, (0, 0), fx=0.5, fy=0.5)
prevgray = cv2.cvtColor(prevgray, cv2.COLOR_BGR2GRAY)

while True:

    if webcam_stream.stopped is True:
        break
    else:
        img = webcam_stream.read()

    # resize the frame by half
    gray = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 10, 15, 1, 2, 1.2, 0)

    prevgray = gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)

    print(f"{fps:.2f} FPS")

    cv2.imshow('flow', draw_flow(gray, flow))
    cv2.imshow('flow HSV',draw_hsv(flow))
    cv2.imshow("img",cv2.resize(img, (0, 0), fx=0.5, fy=0.5))

    key = cv2.waitKey(2)

    if key == ord('p'):
        webcam_stream.pause()

    if key == ord('q'):
        break


