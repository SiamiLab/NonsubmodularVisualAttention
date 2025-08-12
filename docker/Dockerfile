FROM nvidia/opengl:1.0-glvnd-devel-ubuntu16.04

# Add ROS repo
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    apt-get update && \
    apt-get install -y ros-kinetic-desktop-full

# Source ROS and init rosdep
RUN apt-get update && apt-get install -y python-rosdep && \
    /bin/bash -c "source /opt/ros/kinetic/setup.bash && rosdep init || true && rosdep update"

ENV CERES_VERSION="1.12.0"
ENV CATKIN_WS=/root/visualattention_ws
ENV ROS_DISTRO=kinetic

# set up thread number for building
RUN apt-get update && apt-get install -y \
    nano \
    build-essential \    
    git \
    cmake \
    libatlas-base-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libsuitesparse-dev \
    python-catkin-tools \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-message-filters \
    ros-${ROS_DISTRO}-tf && \
    rm -rf /var/lib/apt/lists/* && \
    # Build and install Ceres
    git clone https://ceres-solver.googlesource.com/ceres-solver && \
    cd ceres-solver && \
    git checkout tags/${CERES_VERSION} && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc --ignore=1) install && \
    rm -rf ../../ceres-solver && \
    mkdir -p $CATKIN_WS/src/VINS-Mono/


# Install zsh and Oh My Zsh
RUN apt-get update && apt-get install -y zsh curl && \
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    chsh -s /bin/zsh

# Install OpenSSH server
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd

# Optional: Set root password
RUN echo 'root:root' | chpasswd

# Allow root login (for testing/dev only — secure differently in prod!)
RUN sed -i 's/^#\?PermitRootLogin[: ]\+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication[: ]\+.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Expose SSH port
EXPOSE 22


RUN cd $CATKIN_WS/src && \
    git clone https://github.com/SiamiLab/NonsubmodularVisualAttention.git

# Build VINS-Mono
WORKDIR $CATKIN_WS
ENV TERM xterm
ENV PYTHONIOENCODING UTF-8
RUN catkin config \
      --extend /opt/ros/$ROS_DISTRO \
      --cmake-args \
        -DCMAKE_BUILD_TYPE=Release && \
    catkin build

# source ROS and workspace. activate ssh in zshrc
RUN echo 'source /opt/ros/kinetic/setup.zsh' >> ~/.zshrc && \
    echo 'source /root/visualattention_ws/devel/setup.zsh' >> ~/.zshrc && \
    echo 'service ssh start' >> ~/.zshrc

