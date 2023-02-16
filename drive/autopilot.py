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
STEER_CENTER = 100
MAX_STEER = 50
engine.stop()
steer.angle = STEER_CENTER
# init camera
cv.startWindowThread()
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FPS, 30)
# init autopilot
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


autopilot = DenseNetwork()
# autopilot.load_state_dict(torch.load("./model2023-02-14-16-28.pth", map_location=torch.device('cpu')))
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
