import time
from adafruit_servokit import ServoKit


kit = ServoKit(channels=8, address=0x40)
print("150")
kit.servo[0].angle = 150  # left bound
time.sleep(2)
print("50")
kit.servo[0].angle = 50  # right bound
time.sleep(2)
print("100")
kit.servo[0].angle = 100
time.sleep(2)
