import sys
import os
from datetime import datetime
import numpy as np
import pygame
from pygame.locals import *
from pygame import event, display, joystick
from adafruit_servokit import ServoKit
from gpiozero import PhaseEnableMotor
import cv2 as cv
import csv


# SETUP
# init engine and steering wheel
engine = PhaseEnableMotor(phase=19, enable=26)
kit = ServoKit(channels=8, address=0x40)
steer = kit.servo[0]
MAX_THROTTLE = 0.32
STEER_CENTER = 90
MAX_STEER = 50
assert MAX_THROTTLE <= 1
steer.angle = 90
# init jotstick controller
display.init()
joystick.init()
print(f"{joystick.get_count()} joystick connected")
js = joystick.Joystick(0)
# init camera
cv.startWindowThread()
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FPS, 30)
# create data storage
image_dir = '/home/pbd0/playground/wham_buggy/train_and_deploy/data/' + datetime.now().strftime("%Y%m%d_%H%M") + '/images/'
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
# init vars
record_data = True
speed, ang = 0., 0.
action = []
frame_count = 0


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
                axval_0 = js.get_axis(0)
                axval_4 = js.get_axis(4)
                speed = -np.clip(axval_4, -MAX_THROTTLE, MAX_THROTTLE)
                if speed > 0:
                    engine.forward(speed)
                elif speed < 0:
                    engine.backward(-speed)
                else:
                    engine.stop()
                ang = STEER_CENTER - MAX_STEER * axval_0
                steer.angle = ang
            action = [speed, ang]
            print(f"engine speed: {speed}, steering angle: {ang}")
        if record_data:
            image = cv.resize(frame, (300, 300))
            cv.imwrite(image_dir + str(frame_count)+'.jpg', image)  # save frame
            label = [str(frame_count)+'.jpg'] + list(action)  # save labels
            label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)
        if cv.waitKey(1) == ord('q'):
            engine.stop()
            engine.close()
            cv.destroyAllWindows()
            break
       
except KeyboardInterrupt:
    engine.stop()
    engine.close()
    cv.destroyAllWindows()
