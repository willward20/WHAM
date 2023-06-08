# WHAM (Senior Design Team 2022-2023)

<img src="https://github.com/willward20/WHAM/blob/main/media/WHAM_car.jpg" width="350"/> <img src="https://github.com/willward20/WHAM/blob/main/media/course.jpg" width="450" />

## Project Overview

Senior undergraduates in the [Engineering Physics program](https://uca.edu/physics/engineering-physics/) at the University of Central Arkansas complete a two-semester Senior Design project before graduation. The 2022-2023 senior team designed a small autonomous vehicle to compete in two robotics competitions: (1) the Autonomous Vehicle Challenge at the 2023 [National Robotics Challenge](https://www.thenrc.org/) (NRC) in Marion, Ohio and (2) the Intelligent Vehicle Challenge at the 2023 [Arkansas Space Grant Consortium](https://arkansasspacegrant.org/) (ASGC) Symposium in Morrilton, Arkansas. To earn points in both competitions, the robot needed to autonomously navigate through an obstacle course in under five minutes by driving around five multi-colored buckets, driving under a small arch, and driving over a ramp. An image of the course layout provided by NRC 2023 contest manual is shown above.

We designed our autonomous system using vision-based machine learning, inspired by the open-source API [Donkey Car](https://docs.donkeycar.com/). To train our vehicle to navigate, we first manually drive the vehicle around the obstacle course using a wireless controller while the vehicle records front-facing camera images and the user's driving commands. Next, the recorded data is transferred to an external laptop that trains a Convolutional Neural Network (CNN). The CNN analyzes input images and predicts the correct driving command (a steering value and a throttle value) based on the data it was trained with. Once the network is trained to an acceptable accuracy, it's transferred back to the vehicle. When the vehicle is deployed in autonomous mode, the car uses the trained CNN to makes its own driving decisions based on new images it captures.  

This repository contains all of the code used to train and deploy our autonomous vehicle. The [config.json](https://github.com/willward20/WHAM/blob/main/train_and_deploy/config.json) file limits the vehicle's maximum throttle and defines the vehicle's steering trim. The [motor.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/motor.py) and [servo.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/servo.py) scripts contain functions for initializing and deploying the vehicle's motor/motor driver and steering servo/PWM board. The [collect_data.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/collect_data.py) script is used to manually drive the vehicle with a wireless controller while collecting steering, throttle, and camera data. The [train.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/train.py) script trains a CNN using PyTorch and generates a .pth autopilot file containing the trained parameters. The [autopilot.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/autopilot.py) script drives the vehicle autonomously using a .pth autopilot file. Both the train.py and autopilot.py access neural network architectures that are defined in the [cnn_network.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/cnn_network.py) file. 

## Competition Rules
- Maximum vehicle size: 24" x 24" x 24"
- Vehicle must be fully autonomous and self-contained
- Everything necessary for the vehicle naviguation, processing, sensing must be attached
- No transmission or communication is allowed, except GPS
- No tether to a laptop or a device is allowed
- Vehicle must be started with a physical button, not a wireless communication
- The event runs regardless of the weather
- Points are earned for clearing each obstacle and crossing the finish line

## Goals
- [x] Set up the raspberri pi and software installation
- [x] Set up Github page
- [x] Order part list
- [x] Mechanical Design
- [x] Document the competition Requirements
- [x] Donkey Car software training and model
- [x] Build neural networks for the wham car
- [x] Compare Performance of Donkey to WHAM! car
- [X] Build Video Stabilizer
- [X] Determine and build program for controller type (keyboard or ps4)

## Important Links 
- [National Robotics Challenge](https://www.thenrc.org/)
- [Arkansas Space Grant Consortium](https://arkansasspacegrant.org/)
- [Donkey Car API](https://docs.donkeycar.com/) (inspired this project)

## Contributors 
- [Austin Miller](https://github.com/amillertime)
- [Colby Hoggard](https://github.com/choggard123)
- [Nadira Amadou](https://github.com/nadira30)
- [Will Ward](https://github.com/willward20)

## Advisor
- [Dr. Lin Zhang](https://github.com/linzhangUCA)
