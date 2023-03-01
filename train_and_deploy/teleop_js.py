##################################################################
# Program Name: teleop_js.py
# Contributors: 
# 
#  
###################################################################

#!/usr/bin/python3
import sys
import os
import cv2 as cv
from adafruit_servokit import ServoKit
# import motor
import pygame
import time
# from gpiozero import LED
import json

from time import time

# SETUP
# load configs
config_path = os.path.join(sys.path[0], "config.json")
f = open(config_path)
data = json.load(f)
steering_trim = data['steering_trim']
throttle_lim = data['throttle_lim']
# init servo controller
kit = ServoKit(channels=16)
servo = kit.servo[15]
# init controller
pygame.display.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()
# init camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 20)
# init variables
# led = LED(4)
for _ in reversed(range(60)):
    ret, frame = cap.read()
# init timer
start_stamp = time()
frame_counts = 0
ave_frame_rate = 0.

try:
    while True:
        ret, frame = cap.read()
        # pygame.event.pump()
        throttle = -round((pygame.joystick.Joystick(0).get_axis(1)), 2)  # throttle input: -1: max forward, 1: max backward
        # motor.drive(throttle * throttle_lim)  # apply throttle limit
        steer = round((pygame.joystick.Joystick(0).get_axis(3)), 2)  # steer_input: -1: left, 1: right
        # ang = 90 * (1 + steer) + steering_trim
        # if ang > 180:
        #     ang = 180
        # elif ang < 0:
        #     ang = 0
        # servo.angle = ang
        action = [steer, throttle]
        # print(f"action: {action}")
        frame_counts += 1
        duration_since_start = time() - start_stamp
        ave_frame_rate = frame_counts / duration_since_start
        print(f"frame rate: {ave_frame_rate}")
        if cv.waitKey(1)==ord('q'):
            motor.kill()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
except KeyboardInterrupt:
    motor.kill()
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()
