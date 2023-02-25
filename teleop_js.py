import sys
import numpy as np
import pygame
from pygame.locals import *
from pygame import event, display, joystick
from adafruit_servokit import ServoKit
from gpiozero import PhaseEnableMotor
import cv2 as cv


# SETUP
# init engine and steering wheel
engine = PhaseEnableMotor(phase=19, enable=26)
kit = ServoKit(channels=16, address=0x40)
steer = kit.servo[15]
MAX_THROTTLE = 0.50
STEER_CENTER = 95.5
MAX_STEER = 60
engine.stop()
steer.angle = STEER_CENTER
# init jotstick controller
display.init()
joystick.init()
print(f"{joystick.get_count()} joystick connected")
js = joystick.Joystick(0)
# init camera
cv.startWindowThread()
cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FPS, 20)


# MAIN
try:
    while True:
        ret, frame = cam.read()
        if not ret:  # check camera
            print("No image received!")
            engine.stop()
            engine.close()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()
        for e in event.get():
            if e.type == QUIT:
                print("QUIT detected, terminating...")
                engine.stop()
                engine.close()
                cv.destroyAllWindows()
                pygame.quit()
                sys.exit()
            if e.type == JOYAXISMOTION:
                ax0_val = js.get_axis(0)
                ax4_val = js.get_axis(4)
                vel = -np.clip(ax4_val, -MAX_THROTTLE, MAX_THROTTLE)
                if vel > 0:  # drive motor
                    engine.forward(vel)
                elif vel < 0:
                    engine.backward(-vel)
                else:
                    engine.stop()
                ang = STEER_CENTER + MAX_STEER * ax0_val
                steer.angle = ang  # drive servo
                action = (ax0_val, ax4_val)  # steer, throttle
                print(f"throttle axis: {ax4_val}, steering axis: {ax0_val}\nengine speed: {vel}, steering angle: {ang}")
        if cv.waitKey(1) == ord('q'):
            engine.stop()
            engine.close()
            cv.destroyAllWindows()
            pygame.quit()
            sys.exit()

except KeyboardInterrupt:
    engine.stop()
    engine.close()
    cv.destroyAllWindows()
    pygame.quit()
    sys.exit()
