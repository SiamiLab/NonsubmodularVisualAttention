Non-submodular Visual Attention
--

This project presents the implementation of a task-oriented computational framework to enhance Visual-Inertial Navigation (VIN) in robots, addressing challenges such as limited time and energy resources. The framework strategically selects visual features using a Mean Square Error (MSE)-based, non-submodular objective function and a simplified dynamic anticipation model. To address the NP‐hardness of this problem, we implemented four polynomial‐time approximation algorithms: a classic greedy method with constant‐factor guarantees; a low‐rank greedy variant that significantly reduces computational complexity; a randomized greedy sampler that balances efficiency and solution quality; and a linearization‐based selector based on a first‐order Taylor expansion for near‐constant‐time execution.


https://github.com/user-attachments/assets/ec2f0679-9618-47d6-b84b-63d9494077a1



## Dataset
The dataset from the cancer-ribbon experiment used in this work is available in EuRoC format, provided as both an ASL dataset and a ROS bag. You can access it from [this link](https://northeastern-my.sharepoint.com/:f:/g/personal/behzad_k_northeastern_edu/Eh4IJCtMy25Gulpw5cgLD8YBT8u4B7enzvno0VMq5WsseQ?e=K3e0Fw). (The link will be made publicly available upon acceptance.)



## Installation

This implementation is based on the [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) and [Anticipated-VINS-Mono](https://github.com/plusk01/Anticipated-VINS-Mono) projects. Please refer to `VINS-Mono` project to install the requirements first. Then clone or download this project and execute the following commands.

```bash
# build the workspace
$ catkin build
```

```bash
# run the VIO pipeline
$ roslaunch vins_estimator euroc.launch sequence_name:=stereo_vio_exp_ccw_020_future_horizon_gt
```

```bash
# play the cancer-ribbon experiment bag file
rosbag play /path/to/bag/file.bag --clock
```

```bash
# execute launch file for rviz vizualization
$ roslaunch vins_estimator vins_rviz.launch
```


