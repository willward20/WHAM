import sys
import os
from datetime import datetime
import pygame
from pygame.locals import *
from pygame import event, display, joystick
from adafruit_servokit import ServoKit
from gpiozero import PhaseEnableMotor
import cv2 as cv
import csv
import time


# SETUP
# init engine and steering wheel
engine = PhaseEnableMotor(phase=19, enable=26)
kit = ServoKit(channels=8, address=0x40)
steer = kit.servo[0]
CONST_THROTTLE = 0.2
STEER_CENTER = 87
MAX_STEER = 50
engine.stop()
steer.angle = STEER_CENTER
# init jotstick controller
display.init()
joystick.init()
print(f"{joystick.get_count()} joystick connected")
js = joystick.Joystick(0)
# init camera
cv.startWindowThread()
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FPS, 20)
time.sleep(4)  # to warm up camera
print("Camera is ready...\n")
# create data storage
image_dir = '/home/pbd0/playground/wham_buggy/train/data/' + datetime.now().strftime("%Y%m%d_%H%M") + '/images/'
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
# init vars
record_data = True
vel, ang = 0., 0.
frame_count = 0


# MAIN
try:
    while True:
        ret, frame = cam.read()
        if ret:  # check camera
            frame_count += 1
        else:
            print("No image received!")
            engine.stop()
            engine.close()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
        for e in event.get():
            if e.type == QUIT:
                print("QUIT detected, terminating...")
                engine.stop()
                engine.close()
                cv.destroyAllWindows()
                pygame.quit()
                sys.exit()
            if e.type == JOYAXISMOTION:
                ax0_val = js.get_axis(0)
                ang = STEER_CENTER - MAX_STEER * ax0_val
            else:
                ang = STEER_CENTER
        engine.forward(CONST_THROTTLE)  # drive motor
        steer.angle = ang   # drive servo
        print(f"engine speed: {vel}, steering angle: {ang}")
        if record_data:
            image = cv.resize(frame, (300, 300))
            cv.imwrite(image_dir + str(frame_count)+'.jpg', image)  # save image
            label = [str(frame_count)+'.jpg'] + [ang]
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)  # save labels
        if cv.waitKey(1) == ord('q'):
            engine.stop()
            engine.close()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()

except KeyboardInterrupt:
    engine.stop()
    engine.close()
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()
