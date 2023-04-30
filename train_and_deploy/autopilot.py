
# Deploy trained neural network for self driving 

#!/usr/bin/python3
import sys
import os
import cv2 as cv
from adafruit_servokit import ServoKit
import motor
from gpiozero import LED
import json
import csv
from datetime import datetime

import pyrealsense2 as rs
from time import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import cnn_network

# Check to see if startup is working
from os.path import expanduser
import datetime

file = open(expanduser("~") + '/i_was_created.txt', 'w')
file.write("This LinuxShellTips Tutorial Actually worked!\n" + str(datetime.datetime.now()))
file.close()



## realsense
# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 160, 120, rs.format.rgb8, 30)
else:
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)

# Start streaming
profile = pipeline.start(config)
profile.get_device().query_sensors()[1].set_option(rs.option.auto_exposure_priority, 0.0)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 13.74 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
#realsense


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

model_path = os.path.join(sys.path[0], 'models', 'DonkeyNet_2023_04_11_11_35_15epochs_lr_1E-3.pth')
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

# create data storage
#image_dir = os.path.join(sys.path[0], 'data', datetime.now().strftime("%Y_%m_%d_%H_%M"), 'images/')
#if not os.path.exists(image_dir):
#    try:
#        os.makedirs(image_dir)
#    except OSError as e:
#        if e.errno != errno.EEXIST:
#            raise
#label_path = os.path.join(os.path.dirname(os.path.dirname(image_dir)), 'labels.csv')


# init variables
throttle, steer = 0., 0.
is_recording = False
frame_counts = 0
# init timer, uncomment if you are cuious about frame rate
start_stamp = time()
ave_frame_rate = 0.



# MAIN
try:
    while True:
        frames = pipeline.wait_for_frames()
        if frames is not None:
            frame_counts += 1
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
            ##lines below will blur out background after a given distance, which is around 13.76m for us (max distance between buckets)
            #grey_color = 153
            #depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            #bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
            depth_image = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        else:
            motor.kill()
            head_led.off()
            tail_led.off()
            cv.destroyAllWindows()
            sys.exit()
            
            
            
        # predict steer and throttle
        #print(f"data type of color: ${color_image.dtype} \ndata type of depth: ${depth_image.dtype}")
        color_image = cv.resize(color_image, (120,160))
        depth_image = cv.resize(depth_image, (120, 160))
        color_tensor = to_tensor(color_image)
        #print(color_image.shape, depth_image.shape)
        depth_tensor = to_tensor(depth_image)
        #depth_image = (depth_image / 256).astype(np.uint8)
        #depth_tensor = torch.from_numpy(depth_image)
        #print(color_tensor.shape, depth_tensor.shape)

        pred_steer, pred_throttle = model(color_tensor[None, :], depth_tensor[None, :]).squeeze()
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
        #print(f"action: {action}")
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
