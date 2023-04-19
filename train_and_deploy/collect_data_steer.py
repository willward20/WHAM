
# Collect training data using bluetooth controller
# While driving, save images and joystick inputs

#!/usr/bin/python3
import sys
import os
import cv2 as cv
from adafruit_servokit import ServoKit
import motor
import pygame
from gpiozero import LED
import json
import csv
from datetime import datetime

from time import time


# SETUP
# dummy video driver
os.environ["SDL_VIDEODRIVER"] = "dummy"
# load configs
config_path = os.path.join(sys.path[0], "config.json")
f = open(config_path)
data = json.load(f)
steering_trim = -1 * data['steering_trim']
throttle_percent = data['throttle_percent']
# init servo controller
kit = ServoKit(channels=16)
servo = kit.servo[0]
# init LEDs
head_led = LED(16)
tail_led = LED(12)
# create data storage
image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y_%m_%d_%H_%M"), 'images/')
if not os.path.exists(image_dir):
    try:
        os.makedirs(image_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
# init controller
pygame.display.init()
pygame.joystick.init()
js = pygame.joystick.Joystick(0)
# init variables
steer = 0.
is_recording = False
frame_counts = 0
# init camera
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FPS, 20)
for i in reversed(range(60)):  # warm up camera
    if not i % 20:
        print(i/20)
    ret, frame = cap.read()
# init timer, uncomment if you are cuious about frame rate
start_stamp = time()
ave_frame_rate = 0.
start_time=datetime.now().strftime("%Y_%m_%d_%H_%M_")


# MAIN
try:
    while True:
        ret, frame = cap.read()
        if frame is not None:
            frame_counts += 1
        else:
            motor.kill()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
        for e in pygame.event.get():
            if e.type == pygame.JOYAXISMOTION:
                steer = -js.get_axis(3)  # steer_input: -1: left, 1: right
            elif e.type == pygame.JOYBUTTONDOWN:
                if pygame.joystick.Joystick(0).get_button(0):
                    is_recording = not is_recording
                    head_led.toggle()
                    tail_led.toggle()
                    if is_recording:
                        print("Recording data")
                    else:
                        print("Stopping data logging")
        ang = 90 * (1 + steer) + steering_trim
        if ang > 180:
            ang = 180
        elif ang < 0:
            ang = 0
        servo.angle = ang
        motor.drive(throttle_percent)  # apply throttle limit
        print(f"steer axis value: {steer}, throttle: {throttle_percent}")
        if is_recording:
            frame = cv.resize(frame, (120, 160))
            cv.imwrite(image_dir + start_time+str(frame_counts)+'.jpg', frame) # changed frame to gray
            # save labels
            label = [start_time+str(frame_counts)+'.jpg'] + [steer]
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)  # write the data
        # monitor frame rate
        # duration_since_start = time() - start_stamp
        # ave_frame_rate = frame_counts / duration_since_start
        # print(f"frame rate: {ave_frame_rate}")
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
