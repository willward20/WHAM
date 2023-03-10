##################################################################
# Program Name: autopilot.py
# Contributors: 
# 
# Deploy trained neural network for self driving 
###################################################################


#!/usr/bin/python3
import sys
import os
import cv2 as cv
<<<<<<< HEAD
import motor
from torchvision.transforms import ToTensor, Resize
import torch
import cnn_network
=======
>>>>>>> f1895eb38957ffd1f482227c9af79b61807c156b
from adafruit_servokit import ServoKit
import motor
from gpiozero import LED
import json

from time import time
import torch
import torch.nn as nn
from torchvision import transforms


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
# load model
class DonkeyNetwork(nn.Module):
    """
    Input image size: (120, 160, 3)
    """
    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(64*8*13, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv24(x))  # (120-5)/2+1=58, (160-5)/2+1=78
        x = self.relu(self.conv32(x))  # (58-5)/2+1=27, (78-5)/2+1=37
        x = self.relu(self.conv64_5(x))  # (27-5)/2+1=12, (37-5)/2+1=17
        x = self.relu(self.conv64_3(x))  # 12-3+1=10, 17-3+1=15
        x = self.relu(self.conv64_3(x))  # 10-3+1=8, 15-3+1=13
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


<<<<<<< HEAD
# Load CNN model
model = cnn_network.donkey_net()
model.load_state_dict(torch.load("./donkeynet32epoch_data2023-02-15-17-04.pth", map_location=torch.device('cpu')))

# Setup Transforms
img2tensor = ToTensor()
resize = Resize(size=(300,300))

# Create video capturer
cap = cv.VideoCapture(0) #video capture from 0 or -1 should be the first camera plugged in. If passing 1 it would select the second camera
# cap.set(cv.CAP_PROP_FPS, 10)

times = [] # array to hold the elapsed time between each recieved frame
start_time = time.time()

while True:
    ret, frame = cap.read()   
    if frame is not None:
        #cv.imshow('frame', frame)  # debug
        frame = cv.resize(frame, (int(frame.shape[1]), int(frame.shape[0])))
        img_tensor = img2tensor(frame) # added thi sline to get the colored img 
        #print(f"frame size: {frame.shape}")
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # we changed this from b&w to color
        #img_tensor = img2tensor(gray)
        img_tensor = resize(img_tensor) # I am 90% certain that this line is not needed -- changing the size should not affect predictions
        #print(img_tensor.shape)
    with torch.no_grad():
        pred = model(img_tensor.unsqueeze(dim=0)) # This line adds an extra dimension to the image tensor (print shape before and after to observe this effect)
    #print(pred)
    steering, throttle = pred[0][0].item(), pred[0][1].item()
    if throttle * throttle_lim < -100:
        throttle = -1
    #print("steering: ", steering, "     throttle: ", throttle)
    motor.drive(throttle * throttle_lim) 
    #print("motor: ", throttle * throttle_lim) 
    ang = 90 * (1 + steering) + steering_trim 
    if ang > 180:
        ang = 180
    elif ang < 0:
        ang = 0
    kit.servo[0].angle = ang
    #print("ang: ", ang)

    elapsed_time = time.time() - start_time
    times.append(elapsed_time)
    start_time = time.time()
    #print("elapsed time: ", elapsed_time)
    print("Average Recieved Image Rate: ", len(times) / sum(times))

    if cv.waitKey(1)==ord('q'):
        motor.stop()
        motor.close()
        break
=======
model_path = os.path.join(sys.path[0], 'models', 'donkey16epoch_202303031421_lsc114.pth')
to_tensor = transforms.ToTensor()
model = DonkeyNetwork()
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
>>>>>>> f1895eb38957ffd1f482227c9af79b61807c156b

print(len(times) / sum(times))

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
        motor.drive(throttle * throttle_lim)  # apply throttle limit
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
        print(f"frame rate: {ave_frame_rate}")
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
