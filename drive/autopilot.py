import sys
import os
from datetime import datetime
import numpy as np
from adafruit_servokit import ServoKit
from gpiozero import PhaseEnableMotor
import cv2 as cv

import torch
import torch.nn as nn
from torchvision import transforms



# SETUP
# init engine and steering wheel
engine = PhaseEnableMotor(phase=19, enable=26)
kit = ServoKit(channels=8, address=0x40)
steer = kit.servo[0]
MAX_THROTTLE = 0.32
STEER_CENTER = 87
MAX_STEER = 50
engine.stop()
steer.angle = STEER_CENTER
# init camera
cv.startWindowThread()
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FPS, 30)
# init autopilot
class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1 = nn.Linear(64*18*18, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.conv24(x))  # (200-5)/2+1 = 98
        x = self.relu(self.conv32(x))  # (98-5)/2+1 = 47
        x = self.relu(self.conv64_5(x))  # (47-5)/2+1 = 22
        x = self.relu(self.conv64_3(x))  # 22-3+1 = 20
        x = self.relu(self.conv64_3(x))  # 20-3+1 = 18
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200*200*3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


autopilot = ConvNetwork()
autopilot.load_state_dict(torch.load("/home/pbd0/playground/wham_buggy/train/models/conv_soccer-20230217_1426.pth", map_location=torch.device('cpu')))
to_tensor = transforms.ToTensor()


# MAIN
try:
    while True:
        ret, frame = cam.read()
        if not ret:  # check camera
            print("No image received!")
            engine.stop()
            engine.close()
            cv.destroyAllWindows()
            sys.exit()
        image = cv.resize(frame, (200, 200))
        im_tensor = to_tensor(image)
        pred_ax4, pred_ax0 = autopilot(im_tensor[None, :]).squeeze()
        vel = -np.clip(float(pred_ax4), -MAX_THROTTLE, MAX_THROTTLE)
        if vel > 0:  # drive motor
            engine.forward(vel)
        elif vel < 0:
            engine.backward(-vel)
        else:
            engine.stop()
        ang = STEER_CENTER - MAX_STEER * float(pred_ax0)
        ang = np.clip(ang, STEER_CENTER - MAX_STEER, STEER_CENTER + MAX_STEER)
        steer.angle = ang  # drive servo
        print(f"engine speed: {vel}, steering angle: {ang}")
        if cv.waitKey(1) == ord('q'):
            engine.stop()
            engine.close()
            cv.destroyAllWindows()
            sys.exit()

except KeyboardInterrupt:
    engine.stop()
    engine.close()
    cv.destroyAllWindows()
    sys.exit()
