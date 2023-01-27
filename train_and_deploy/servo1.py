from time import time, sleep
from adafruit_servokit import ServoKit

#pca.frequency = 50
kit = ServoKit(channels=16)
calibrate = 7

def right(angle):
    kit.servo[0].angle = 90 + angle
def left(angle):
    kit.servo[0].angle = 90 + angle 
def reset():
    kit.servo[0].angle = 90 + calibrate
def turn(deg):
    angle = 90 + deg * 90 + calibrate
    if angle > 180:
        angle = 180
    elif angle < 0:
        angle = 0
    kit.servo[0].angle = angle
