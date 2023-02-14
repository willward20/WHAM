import sys
import os
from datetime import datetime
import time
import numpy as np
import pygame
from pygame.locals import *
from pygame import event, display, joystick
from adafruit_servokit import ServoKit
from gpiozero import PhaseEnableMotor
import cv2 as cv
import csv


# SETUP
engine = PhaseEnableMotor(phase=19, enable=26)
kit = ServoKit(channels=8, address=0x40)
steer = kit.servo[0]
MAX_THROTTLE = 0.25
STEER_CENTER = 100
MAX_STEER = 50
assert MAX_THROTTLE <= 1
steer.angle = 90
# init vars
record_data = True
speed, ang = 0., 0.
action = []
frame_count = 0

# init controller
display.init()
joystick.init()
print(f"{joystick.get_count()} joystick connected")
js = joystick.Joystick(0)
# init camera
cv.startWindowThread()
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FPS, 30)
# create data storage
image_dir = 'data' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/images/'
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# MAIN
try:
    while True:
        ret, frame = cam.read()
        frame_count += 1
        for e in event.get():
            if e.type == QUIT:
                print("QUIT detected, terminating...")
                pygame.quit()
                sys.exit()
            if e.type == JOYAXISMOTION:
                v_0 = js.get_axis(0)
                v_4 = js.get_axis(4)
                # print(f"steering joystick value: {v_0}, speed joystick value: {v_4}")
                speed = -np.clip(v_4, -MAX_THROTTLE, MAX_THROTTLE)
                if speed > 0:
                    engine.forward(speed)
                elif speed < 0:
                    engine.backward(-speed)
                else:
                    engine.stop()
                ang = STEER_CENTER - MAX_STEER * v_0
                steer.angle = ang
            action = [ang, speed]
        if record_data:
            image = cv.resize(frame, (300, 300))
            cv.imwrite(image_dir + str(frame_count)+'.jpg', image)  # save frame
            # save labels
            label = [str(frame_count)+'.jpg'] + list(action)
            label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)  # write the data
        if cv.waitKey(1) == ord('q'):
            engine.stop()
            engine.close()
            cv.destroyAllWindows()
            break
       
except KeyboardInterrupt:
    engine.stop()
    engine.close()
    cv.destroyAllWindows()
