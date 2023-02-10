from setuptools import setup

setup(
    name='wham_buggy',
    version='0.0.1',
    install_requires=[
        'numpy>=1.23',
        'opencv-python==4.5.5.64',
        'pygame==2.1.2',
        'matplotlib>=3.6.2',
        'adafruit_circuitpython_servokit',
        'gpiozero',
    ]
)
