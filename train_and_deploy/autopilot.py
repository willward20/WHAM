
# Deploy trained neural network for self driving 


#!/usr/bin/python3
import sys
import os
import cv2 as cv
from adafruit_servokit import ServoKit
import motor
# import pygame
from gpiozero import LED
import json
from datetime import datetime

from time import time
# from torchvision.transforms import ToTensor, Resize
# import torch
# import torch.nn as nn
import torch
import torch.nn as nn
from torchvision import transforms

os.environ["SDL_VIDEODRIVER"] = "dummy"
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
<<<<<<< HEAD
# load model
model_path = os.path.join(sys.path[0], 'models', 'donkey32epoch_202303031347_volleyball.pth')
# img2tensor = ToTensor()
to_tensor = transforms.ToTensor()
model = DonkeyNetwork()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# init controller
# pygame.display.init()
# pygame.joystick.init()
# js = pygame.joystick.Joystick(0)
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


# MAIN
try:
    while True:
        ret, frame = cap.read()
        if frame is not None:
            frame_counts += 1
            # frame = cv.resize(frame, (120, 160))
            # img_tensor = img2tensor(frame) # added thi sline to get the colored img 
            image = cv.resize(frame, (120, 160))
            img_tensor = to_tensor(image)

        else:
            motor.kill()
            cv.destroyAllWindows()
            # pygame.quit()
            sys.exit()
        # for e in pygame.event.get():
            # if e.type == pygame.JOYAXISMOTION:
            #     throttle = -round((js.get_axis(1)), 2)  # throttle input: -1: max forward, 1: max backward
            #     steer = -1 * round((js.get_axis(3)), 2)  # steer_input: -1: left, 1: right
            # elif e.type == pygame.JOYBUTTONDOWN:
            #     if pygame.joystick.Joystick(0).get_button(0):
            #         is_recording = not is_recording
            #         head_led.toggle()
            #         tail_led.toggle()
            #         if is_recording:
            #             print("Recording data")
            #         else:
            #             print("Stopping data logging")
        # with torch.no_grad():
        #     pred = model(img_tensor.unsqueeze(dim=0)) # This line adds an extra dimension to the image tensor (print shape before and after to observe this effect)
        # steer, throttle = pred[0][0].item(), pred[0][1].item()
        steer, throttle = model(img_tensor[None, :]).squeeze()
        steer = round(float(steer), 2)
        throttle = round(float(throttle), 2)
        if throttle >= 1:
            throttle = .999
        elif throttle <= -1:
            throttle = -.999
        motor.drive(throttle * throttle_lim)  # apply throttle limit
=======


model_path = os.path.join(sys.path[0], 'models', 'MODEL.pth')
to_tensor = transforms.ToTensor()
model = cnn_network.DonkeyNetwork()
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
>>>>>>> 5163518cbbd44c7afe01fd49d8e958250505ccfc
        ang = 90 * (1 + steer) + steering_trim
        if ang > 180:
            ang = 180
        elif ang < 0:
            ang = 0
        servo.angle = ang
        action = [steer, throttle]
        print(f"action: {action}")
<<<<<<< HEAD
        # if is_recording:
        #     frame = cv.resize(frame, (120, 160))
        #     cv.imwrite(image_dir + str(frame_counts)+'.jpg', frame) # changed frame to gray
        #     # save labels
        #     label = [str(frame_counts)+'.jpg'] + action
        #     with open(label_path, 'a+', newline='') as f:
        #         writer = csv.writer(f)
        #         writer.writerow(label)  # write the data
=======
>>>>>>> 5163518cbbd44c7afe01fd49d8e958250505ccfc
        # monitor frame rate
        duration_since_start = time() - start_stamp
        ave_frame_rate = frame_counts / duration_since_start
        print(f"frame rate: {ave_frame_rate}")
        if cv.waitKey(1)==ord('q'):
            motor.kill()
            cv.destroyAllWindows()
<<<<<<< HEAD
            # pygame.quit()
            sys.exit()
except KeyboardInterrupt:
    motor.kill()
    cv.destroyAllWindows()
    # pygame.quit()
=======
            head_led.off()
            tail_led.off()
            sys.exit()
except KeyboardInterrupt:
    motor.kill()
    head_led.off()
    tail_led.off()
    cv.destroyAllWindows()
>>>>>>> 5163518cbbd44c7afe01fd49d8e958250505ccfc
    sys.exit()
