# Cancer Ribbon Dataset (ASL & ROS Bag Formats)

This dataset captures a [Quanser QCar](https://www.quanser.com/products/qcar/) ground vehicle equipped with a ZED2 stereo camera performing a cancer ribbon–shaped trajectory in 2D within a 6 × 6 m area. Ground truth is recorded using 8 Motive motion capture cameras mounted overhead.

The dataset is provided in two formats: **ROS bag** and **ASL folder structure**.  
Both follow the well-known **EuRoC MAV dataset format**, allowing you to use them directly with any pipeline or toolbox that supports EuRoC datasets.

- **ASL format**: Organized exactly like the official EuRoC datasets, including timestamps, calibration files, and sensor data.
- **ROS bag**: Contains the same stereo camera and IMU data, along with additional control input signals (throttle and steering) from the Quanser QCar.

To use the dataset, simply load the desired format into your SLAM or Visual-Inertial Navigation framework as you would with standard EuRoC data.

In addition, the dataset also includes calibration data:

- **Camera calibration**: Checkerboard images (left/right lenses) for estimating the intrinsic parameters of the onboard cameras.
- **IMU calibration**: Stationary IMU measurements (~284 seconds) for estimating noise parameters.


## File Downloads
You can download the dataset from [IEEE DataPort](https://dx.doi.org/10.21227/n36c-xz38) *(link will be made publicly available upon paper acceptance)*.
- `cancer_ribbon_ASL.zip` – Dataset in ASL folder format.  
- `cancer_ribbon_bag.zip` – Dataset in ROS bag format.
- `camera_calibration_images.zip` – Camera calibration images.  
- `imu_calibarion_data.zip` – IMU sensor calibration data.


## ASL Format


```txt
cancer_ribbon_ASL
└── qcar0
    ├── body.yaml
    ├── cam0
    │   ├── data
    │   │   ├── 10025405600.png
    │   │   ├── 10065473280.png
    │   │   
    │   ├── data.csv
    │   └── sensor.yaml
    ├── cam1
    │   ├── data
    │   │   ├── 10025405600.png
    │   │   ├── 10065473280.png
    │   │   
    │   ├── data.csv
    │   └── sensor.yaml
    ├── imu0
    │   ├── data.csv
    │   └── sensor.yaml
    └── mocap0
        ├── data.csv
        └── sensor.yaml
```



### qcar0/
Main directory containing all sensor recordings for a single run of the Quanser QCar platform.

- `body.yaml` – General robot information.

---

### cam0/
Contains the left stereo camera recordings.
- `data/` – PNG images representing recorded frames.  
  - File names are timestamps in nanoseconds.
- `data.csv` – Lists frame filenames and their corresponding timestamps (ns).
- `sensor.yaml` – Intrinsic and extrinsic parameters of the left lens.

---

### cam1/
Contains the right stereo camera recordings. (Same structure as `cam0`)
- `data/` – PNG images, filenames = timestamps (ns).
- `data.csv` – Frame filename and timestamp (ns).
- `sensor.yaml` – Intrinsic and extrinsic parameters of the right lens.

---

### imu0/
Contains synchronized inertial sensor recordings.
- `data.csv` – IMU readings over time:  
  - Timestamp (ns)  
  - Angular velocity (3-axis, rad/s)  
  - Linear acceleration (3-axis, m/s²)  
- `sensor.yaml` – IMU extrinsics and noise model parameters.

---

### mocap0/
Contains ground truth robot pose data recorded using a motion capture system.
- `data.csv` – Motion capture measurements over time:  
  - Timestamp (ns)  
  - Position (x, y, z)  
  - Orientation as a quaternion (x, y, z, w)
- `sensor.yaml` – Extrinsics of the motion capture coordinate frame.


## ROS bag Format

```txt
Start time (epoch): 0.000000 s
End time   (epoch): 46.499933 s
Duration: 46.500 s

Topics:
  /cam0/image_raw                           [sensor_msgs/msg/Image]  msgs: 1306
  /cam1/image_raw                           [sensor_msgs/msg/Image]  msgs: 1306
  /imu0                                     [sensor_msgs/msg/Imu]  msgs: 2325
  /user_command                             [geometry_msgs/msg/Vector3Stamped]  msgs: 464
  /velocity_encoder                         [geometry_msgs/msg/Vector3Stamped]  msgs: 2323
```


- `/cam0/image_raw`
Publishes images from the left stereo camera along with their capture timestamp.  

- `/cam1/image_raw`
Publishes images from the right stereo camera along with their capture timestamp.  

- `/imu0`
Publishes timestamped IMU measurements:  
    - Angular velocity (3-axis, rad/s)  
    - Linear acceleration (3-axis, m/s²)  

- `/user_command`
Publishes control inputs sent to the robot over time.  
Vector3 components:  
    - `x`: PWM percentage applied to the motor (driving speed)  
    - `y`: Steering angle in radians  
    - `z`: Unused

- `/velocity_encoder`
Publishes velocity measurements from the wheel encoders.  
Vector3 components:  
    - `x`: Velocity in the x-direction (m/s)  
    - `y`: Velocity in the y-direction (m/s)  
    - `z`: Unused  


## Camera Calibration Images

Although the ASL format already includes our calibrated parameters, these images are provided in case you wish to perform your own calibration. This file contains checkerboard images captured with the stereo camera on the robot. The images are intended for camera calibration and estimation of the intrinsic parameters of the stereo pair.

File format:
  - time_left.jpg  $\rightarrow$  image from the left camera
  - time_right.jpg $\rightarrow$  image from the right camera

Checkerboard properties:
  - Board size: 10 x 7 squares
  - Inner corners: 9 x 6
  - Square size: 0.039 m (width) × 0.037 m (height)


## IMU Sensor Calibration Data

Although the ASL format already includes our calibrated parameters, this data is provided in case you wish to perform your own calibration. This file contains IMU measurements recorded while the robot was stationary (~284 seconds). The dataset can be used to estimate noise parameters of the onboard IMU sensor (e.g., bias instability, random walk, and measurement noise characteristics).

The CSV file includes timestamped IMU measurements. columns are:
  - timestamp [ns]
  - angular_velocity_x [rad/s]
  - angular_velocity_y [rad/s]
  - angular_velocity_z [rad/s]
  - linear_acceleration_x [m/s^2]
  - linear_acceleration_y [m/s^2]
  - linear_acceleration_z [m/s^2]
