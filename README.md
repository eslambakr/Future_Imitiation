# Project hierarchy:
1- Imitation_Learning: Training and testing agent which aim to drive the vehicle autonomously by outputing the vehicle's actions (Throttle / Steering Angle / Brake)

2- video_prediction: Training and Testing agent to predict future frames given the current and previous ones.

# Project Requirments:
1- Install CARLA binary version 0.8.4

2- Install all the python packages listed in requirements.txt

3- Download the training datasets from :https://drive.google.com/file/d/1hloAeyamYn-H6MfV1dRtY1gJPhkR55sY/view

# Dataset describtion:
The data is stored on HDF5 files. Each HDF5 file contains 200 data points. The HDF5 contains two "datasets": 'images_center': 
The RGB images stored at 200x88 resolution

'targets': 
All the controls and measurements collected. They are stored on the "dataset" vector.

1- Steer, float

2- Gas, float

3-Brake, float

3- Hand Brake, boolean

4- Reverse Gear, boolean

5- Steer Noise, float

6- Gas Noise, float

7- Brake Noise, float

8- Position X, float

9- Position Y, float

10- Speed, float
11- Collision Other, float

12- Collision Pedestrian, float

13- Collision Car, float

14- Opposite Lane Inter, float

15- Sidewalk Intersect, float

16- Acceleration X,float

17- Acceleration Y, float

18- Acceleration Z, float

19- Platform time, float

20- Game Time, float

21- Orientation X, float

22- Orientation Y, float

23- Orientation Z, float

24- High level command, int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)

25- Noise, Boolean ( If the noise, perturbation, is activated, (Not Used) )

26- Camera (Which camera was used)

27- Angle (The yaw angle for this camera)
