"""
This module is supposed to work with Cytron MD20A motor driver and an
ARRMA MEGA 550 12T brushed DC motor.
Wiring:
    MOTOR: RED   -> DRIVER: MA
    MOTOR: BLACK -> DRIVER: MB
"""
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# AN1 = 26
# DIG1 = 19
PWM_PIN = 26
DIR_PIN = 19

GPIO.setup(PWM_PIN, GPIO.OUT)
GPIO.setup(DIR_PIN, GPIO.OUT)

# p1 = GPIO.PWM(AN1, 100)
pwm = GPIO.PWM(PWM_PIN, 1000)

# def forward(speed):
#     GPIO.output(DIR_PIN, GPIO.LOW)
#     pwm.start(speed)
#
# def backward(speed):
#     GPIO.output(DIR_PIN, GPIO.HIGH)
#     pwm.start(speed)

def drive(speed):
    """Motor driving function
    Note: when listen to the joystick, verify the sign of speed.
    Args:
        speed: float in range [-100, 100]
    """
    assert speed > 100
    assert speed < -100
    if speed > 0:
        GPIO.output(DIR_PIN, GPIO.LOW)  # forward
        pwm.start(speed)
    elif speed < 0:
        GPIO.output(DIR_PIN, GPIO.HIGH)  # backward
        speed = -speed
        pwm.start(speed)
    else:
        pwm.start(0)

def stop():
    pwm.start(0)

def kill():
    stop()
    GPIO.cleanup
