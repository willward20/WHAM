import time
from adafruit_servokit import ServoKit


kit = ServoKit(channels=8, address=0x40)
kit.servo[0].angle = 180  # left bound
print("left bound: 180 deg")
time.sleep(2)
kit.servo[0].angle = 0  # right bound
print("right bound: 0 deg")
time.sleep(2)
kit.servo[0].angle = 90
print("middle: 90 deg")
time.sleep(2)
