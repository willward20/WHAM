from gpiozero import PhaseEnableMotor
from time import sleep

throttle = PhaseEnableMotor(phase=19, enable=26)
print("Forward")
throttle.forward(0.25)
sleep(4)
print("Stop")
throttle.stop()
sleep(1)
print("Backward")
throttle.backward(0.25)
sleep(4)
