# WHAM (Senior Design Team 2022-2023)

<img src="https://github.com/willward20/WHAM/blob/main/media/WHAM_car.jpg" width="350"/> <img src="https://github.com/willward20/WHAM/blob/main/media/course.jpg" width="450" />

## Project Overview
Our team designed a small autonomous vehicle to compete in two robotics competitions: (1) the Autonomous Vehicle Challenge at the 2023 [National Robotics Challenge](https://www.thenrc.org/) (NRC) and (2) the Intelligent Vehicle Challenge at the 2023 [Arkansas Space Grant Consortium](https://arkansasspacegrant.org/) (ASGC) Symposium. To earn points in both competitions, the robot needed to autonomously navigate through an obstacle course by driving around five multi-colored bucket waypoints, driving under a small arch, and driving over a ramp. An image of the course layout provided by 2023 NRC contest manual is shown above.

We designed our autonomous system using vision-based machine learning, inspired by the open-source API [Donkey Car](https://docs.donkeycar.com/). To teach our robot to navigate, we first manually drive the vehicle around the obstacle course using a wireless controller. The robot records two sets of data while driving: (1) images captured by a mounted camera and (2) the user's driving commands. After driving the car for 10-15 laps, we optimize a Convolutional Neural Network (CNN) using the recorded data. Through an iterative training process, the CNN identifies patterns between what the robot sees in each image and what driving command the user sends to the vehicle. Then, when the CNN receives a new set of image data, it predicts the driving commands for each image. Once the CNN's predictions reach an acceptable level of accuracy, we test the autopilot algorithm live on the course, allowing the trained network to drive the robot autonomously.

This `train_and_deploy` folder contains all of the software for our autonomous vehicle, including:
- a [config.json](https://github.com/willward20/WHAM/blob/main/train_and_deploy/config.json) file that limits the vehicle's maximum throttle and defines the vehicle's steering trim;
- [motor.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/motor.py) and [servo.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/servo.py) scripts that contain functions for initializing and deploying the vehicle's motor/motor driver and steering servo/PWM board;
- a [collect_data.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/collect_data.py) script that is used to manually drive the vehicle with a wireless controller while collecting steering, throttle, and camera data; 
- a [train.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/train.py) script that trains a CNN using PyTorch and generates a .pth autopilot file containing the trained parameters; 
- an [autopilot.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/autopilot.py) script that drives the vehicle autonomously using a .pth autopilot file; 
- and a [cnn_network.py](https://github.com/willward20/WHAM/blob/main/train_and_deploy/cnn_network.py) file that defines the neural network architectures accessed by the train.py and autopilot.py scripts. 

## Convolutional Neural Network
After testing several variations of CNN architectures, we had the most success with Donkey Car's [fastai](https://github.com/autorope/donkeycar/blob/main/donkeycar/parts/fastai.py) architecture (see the "Linear" class). The figure below shows how we modified the CNN structure to accomodate for our input image size and achieve the best results.  Our network architecture has five convolution layers and three fully connected layers. Each image in the network has an input size of 3x120x160 (3 color channels, 120 pixel horizontal width, and 160 pixel vertical height) and an output size of 2 prediction values: steering and throttle. When a dataset is loaded, the recorded images are split into training images (90-95% of the data) and test images (10-5%). During the training process, the network uses the Mean Square Error (MSE) loss function, Adam optimizer, a learning rate of 1E-3, and batch sizes of 125 (train) and 125 (test). The neural network iteratively trains for typically 10-20 epochs. We found the most success when using datasets with a size between 15-20k images.

<img src="https://github.com/willward20/WHAM/blob/main/media/cnn_architecture.png"/>

## Competition Rules
- Maximum vehicle size: 24" x 24" x 24"
- Vehicle must be fully autonomous and self-contained
- Everything necessary for the vehicle naviguation, processing, sensing must be attached
- No transmission or communication is allowed, except GPS
- No tether to a laptop or a device is allowed
- Vehicle must be started with a physical button, not a wireless communication
- The event runs regardless of the weather
- Points are earned for clearing each obstacle and crossing the finish line

## Autonomous Vehicle Performance
**I'm going to add some YouTube video links here to show how the performance of the car varied due to weather conditions and training parameters.**
- videos from before the competition (outside on rainy days, and inside models)
- videos from NRC (outdoors fail)
- videos from the ASGC challenge (winning run, and a bad run too) -- signficant to note that (1) the competition was shaded and (2) we did not train with the yellow bucket. 

## Project Conclusions
We built an autonomous ground vehicle using a modified RC car, a Raspberry Pi, and a vision-based neural network. We can connect a bluetooth joystick with the car to engage the throttle, adjust the steering, and record data for neural network training: camera images and user joystick input. After collecting driving data on a course, we can train the parameters of a convolutional neural network to reduce the loss between predictions and ground truth values. Finally, we can import the trained parameters into a neural network autopilot file that autonomously controls the steering and throttle of the vehicle when camera images are received. The autopilot performs well when operating indoors or under a shaded area outside. However, when driving outside in the sun, the performance accuracy decreased dramatically.

## Future Work
One area of investigation that still needs work is determining the best method for training the neural network. Our most successful tests have been data sets collected continuously, usually with between 15 thousand and 20 thousand images. We would like to see this project taken to the next level, with a neural network model that is general enough to operate under changing conditions (background, weather, time of day) without additional data collection. We tried combining data collected on the course under these different circumstances and training a model, but they performed very poorly. New sensors could potentially help with this. There have been similar projects that use additional information like stereo depth vision and absolute GPS position data based on the start point to make more general models.

While we found a neural network architecture that produced functional models, it was very picky about the weather conditions. We only had successful models when the robot was out of sunlight, and the surroundings were not too bright. This could have been due to the camera itself creating image artifacting from the light, or it could have been because of the model putting too much weight on the brightness of colors. Either way, we recommend trying other neural network architectures to see if there are any that perform better than Donkey Car's fastai architecture.


## Important Links 
- [Successful Autopilot Performance](https://www.youtube.com/watch?v=aOQVNasl_Vw)
- [National Robotics Challenge](https://www.thenrc.org/)
- [Arkansas Space Grant Consortium](https://arkansasspacegrant.org/)
- [Donkey Car API](https://docs.donkeycar.com/) (inspired this project)
- [Donkey Car CNN architecture: fastai](https://github.com/autorope/donkeycar/blob/main/donkeycar/parts/fastai.py)

## Contributors 
We completed this design project during our senior year at the University of Central Arkansas. In May 2023, we each graduated with a Bachelor's of Science in [Engineering Physics](https://uca.edu/physics/engineering-physics/). 
- [Austin Miller](https://github.com/amillertime)
- [Colby Hoggard](https://github.com/choggard123)
- [Nadira Amadou](https://github.com/nadira30)
- [Will Ward](https://github.com/willward20)

## Advisor
- [Dr. Lin Zhang](https://github.com/linzhangUCA) (Thank you for your constant support and enthusiasm!)

## Team Checklist
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
