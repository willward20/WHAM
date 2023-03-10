##################################################################
# Program Name: collect_data.py
# Contributors: 
# 
#  
###################################################################

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
throttle_lim = data['throttle_lim']
# init servo controller
kit = ServoKit(channels=16)
servo = kit.servo[0]
# init LEDs
head_led = LED(16)
tail_led = LED(12)
# create data storage
image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y%m%d%H%M"), 'images/')
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
<<<<<<< HEAD
pygame.joystick.Joystick(0).init()
#stabilizer = VidStab()
cap = cv.VideoCapture(0) #video capture from 0 or -1 should be the first camera plugged in. If passing 1 it would select the second camera
cap.set(cv.CAP_PROP_FPS, 30)
i = 0  # image index
action = [0., 0.]
Record_data = -1
led = LED(4)
headlight = LED(16)
=======
js = pygame.joystick.Joystick(0)
# init variables
throttle, steer = 0., 0.
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
>>>>>>> f1895eb38957ffd1f482227c9af79b61807c156b


<<<<<<< HEAD
    #get thorttle and steering values from joysticks.
    pygame.event.pump()
    throttle = round((pygame.joystick.Joystick(0).get_axis(1)),2) # between -1 (max reverse) and 1 (max forward), rounded to 2 sig figs 
    motor.drive(throttle * throttle_lim) # multiplies speed within range -100 to 100 (or whatever throttle_lim is)
    steer = round((pygame.joystick.Joystick(0).get_axis(3)), 2) # between -1 (left) and 1 (right)
    ang = 90 * (1 + steer) + steering_trim
    if ang > 180:
        ang = 180
    elif ang < 0:
        ang = 0
    kit.servo[15].angle = ang
    # steer = 90 + steering_trim + steer * 90
    # servo.turn(steer)
    # turn(steer)
    print(throttle*throttle_lim, ang)
    
    ##########################################################################################################################################
    action = [steer, throttle] # this MUST be [steering, throttle] because that's the order that train.py expects (originaly it was reversed)
    ##########################################################################################################################################

    # print(f"action: {action}") # debug
    # save image
    if pygame.joystick.Joystick(0).get_button(0) == 1:
        Record_data = Record_data * -1
        if Record_data == 1:
            print("Recording Data")
            led.on()
            headlight.on()
        else:
            print("Stopping Data Logging")
            led.off()
            headlight.off()
        time.sleep(0.1)
    
    if Record_data == 1:
        cv.imwrite(image_dir + str(i)+'.jpg', frame) # changed frame to gray
        # save labels
        label = [str(i)+'.jpg'] + list(action)
        label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')
        with open(label_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(label)  # write the data
        i += 1
        
    if cv.waitKey(1)==ord('q'):
        break
    
        
=======
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
                throttle = -js.get_axis(1)  # throttle input: -1: max forward, 1: max backward
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
        motor.drive(throttle * throttle_lim)  # apply throttle limit
        ang = 90 * (1 + steer) + steering_trim
        if ang > 180:
            ang = 180
        elif ang < 0:
            ang = 0
        servo.angle = ang
        action = [steer, throttle]
        print(f"action: {action}")
        if is_recording:
            frame = cv.resize(frame, (120, 160))
            cv.imwrite(image_dir + str(frame_counts)+'.jpg', frame) # changed frame to gray
            # save labels
            label = [str(frame_counts)+'.jpg'] + action
            with open(label_path, 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(label)  # write the data
        # monitor frame rate
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
>>>>>>> f1895eb38957ffd1f482227c9af79b61807c156b
