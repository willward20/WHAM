import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

#AN2 = 25
AN1 = 26
#DIG2 = 23
DIG1 = 19

#GPIO.setup(AN2, GPIO.OUT)
GPIO.setup(AN1, GPIO.OUT)
#GPIO.setup(DIG2, GPIO.OUT)
GPIO.setup(DIG1, GPIO.OUT)

sleep(1)

p1 = GPIO.PWM(AN1, 100)
#p2 = GPIO.PWM(AN2, 100)

def forward(speed):
    GPIO.output(DIG1, GPIO.LOW)
    #GPIO.output(DIG2, GPIO.HIGH)
    p1.start(speed)
    #p2.start(speed)

def backward(speed):
    GPIO.output(DIG1, GPIO.HIGH)
    #GPIO.output(DIG2, GPIO.LOW)
    p1.start(speed)
    #p2.start(speed)

def drive(speed):
    if speed > 0:
        GPIO.output(DIG1, GPIO.HIGH)
        speed = abs(speed)
        p1.start(speed)
    elif speed < 0:
        GPIO.output(DIG1, GPIO.LOW)
        speed = abs(speed)
        p1.start(speed)
    else:
        p1.start(0)

def stop():
    p1.start(0)
    #p2.start(0)
"""
stop()
GPIO.cleanup()
"""
