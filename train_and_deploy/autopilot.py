
# Deploy trained neural network for self driving 


#!/usr/bin/python3
import sys
import os
import cv2 as cv
from adafruit_servokit import ServoKit
import motor
from gpiozero import LED
import json

from time import time
import torch
import torch.nn as nn
from torchvision import transforms
import cnn_network


# SETUP
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

model_path = os.path.join(sys.path[0], 'models', 'MODEL.pth')
to_tensor = transforms.ToTensor()
model = cnn_network.DonkeyNet()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
head_led.on()
tail_led.on()
# init timer, uncomment if you are cuious about frame rate
start_stamp = time()
ave_frame_rate = 0.


# MAIN
try:
    while True:
        ret, frame = cap.read()
        if frame is not None:
            frame_counts += 1
        else:
            motor.kill()
            head_led.off()
            tail_led.off()
            cv.destroyAllWindows()
            sys.exit()
        # predict steer and throttle
        image = cv.resize(frame, (120, 160))
        img_tensor = to_tensor(image)
        pred_steer, pred_throttle = model(img_tensor[None, :]).squeeze()
        steer = float(pred_steer)
        throttle = float(pred_throttle)
        if throttle >= 1:  # predicted throttle may over the limit
            throttle = .999
        elif throttle <= -1:
            throttle = -.999
        motor.drive(abs(throttle * throttle_lim))  # apply throttle limit
        ang = 90 * (1 + steer) + steering_trim
        if ang > 180:
            ang = 180
        elif ang < 0:
            ang = 0
        servo.angle = ang
        action = [steer, throttle]
        print(f"action: {action}")
        # monitor frame rate
        duration_since_start = time() - start_stamp
        ave_frame_rate = frame_counts / duration_since_start
        #print(f"frame rate: {ave_frame_rate}")
        if cv.waitKey(1)==ord('q'):
            motor.kill()
            cv.destroyAllWindows()
            head_led.off()
            tail_led.off()
            sys.exit()
except KeyboardInterrupt:
    motor.kill()
    head_led.off()
    tail_led.off()
    cv.destroyAllWindows()
    sys.exit()
