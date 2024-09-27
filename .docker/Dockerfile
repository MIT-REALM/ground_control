# FROM osrf/ros:noetic-desktop-full

# # Install packages
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     git \
#     python3-pip \
#     python3-dev \
#     python3-tk \
#     ros-noetic-driver-base \
#     ros-noetic-sophus \
#     ros-noetic-robot-pose-ekf \
#     ros-noetic-ackermann-msgs \
#     ros-noetic-urg-node \
#     ros-noetic-realsense2-camera \
#     ros-noetic-foxglove-bridge

# RUN pip3 install numpy "jax[cpu]" pyyaml rospkg transforms3d equinox scipy matplotlib
# RUN pip3 install --upgrade numpy
# RUN pip3 install torch
# # Install updated cmake
# RUN apt-get update \
#     && apt-get -y install build-essential \
#     && apt-get install -y wget \
#     && rm -rf /var/lib/apt/lists/* \
#     && wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
#     -q -O /tmp/cmake-install.sh \
#     && chmod u+x /tmp/cmake-install.sh \
#     && mkdir /opt/cmake-3.24.1 \
#     && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
#     && rm /tmp/cmake-install.sh \
#     && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

# # Make a ROS workspace
# RUN mkdir -p /catkin_ws/src
# WORKDIR /catkin_ws
# SHELL ["/bin/bash", "-c"]  # so we can use `source`
# RUN source /opt/ros/noetic/setup.sh && catkin_init_workspace src

# # Clone the vicon bridge
# WORKDIR /catkin_ws/src
# RUN git clone https://github.com/ethz-asl/vicon_bridge.git
# # Build
# WORKDIR /catkin_ws
# RUN rosdep install --from-paths src --ignore-src --rosdistro noetic -y
# RUN source /opt/ros/noetic/setup.sh && catkin_make

# # Clone f1tenth system
# WORKDIR /catkin_ws/src
# COPY submodules/f1tenth_system /catkin_ws/src/f1tenth_system

# # Ignore the packages that we don't need (these are only needed to run onboard the car)
# WORKDIR /catkin_ws/src/f1tenth_system
# WORKDIR /catkin_ws/src/f1tenth_system/hokuyo_node
# RUN touch CATKIN_IGNORE
# WORKDIR /catkin_ws/src/f1tenth_system/joystick_drivers
# RUN touch CATKIN_IGNORE
# WORKDIR /catkin_ws/src/f1tenth_system/rl_runtime
# RUN touch CATKIN_IGNORE
# WORKDIR /catkin_ws/src/f1tenth_system/scan2pc
# RUN touch CATKIN_IGNORE
# WORKDIR /catkin_ws/src/f1tenth_system/serial
# RUN touch CATKIN_IGNORE
# WORKDIR /catkin_ws/src/f1tenth_system/vesc/vesc_driver
# RUN touch CATKIN_IGNORE
# WORKDIR /catkin_ws/src/f1tenth_system/vesc/vesc
# RUN touch CATKIN_IGNORE

# # Build
# WORKDIR /catkin_ws
# RUN apt-get update && apt-get install -y 
# RUN rosdep install --from-paths src --ignore-src --rosdistro noetic -y
# RUN source /opt/ros/noetic/setup.sh && catkin_make

# # Add lines to the bashrc file that source ROS
# RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
# RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc

# from sim branch in main repo

FROM osrf/ros:noetic-desktop-full
ARG PYTHON_VERSION=3.9

# Install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    python3-tk \
    python3-netifaces \
    ros-noetic-driver-base \
    ros-noetic-sophus \
    ros-noetic-robot-pose-ekf \
    ros-noetic-ackermann-msgs \
    ros-noetic-urg-node \
    ros-noetic-realsense2-camera \
    ros-noetic-foxglove-bridge \ 
    ros-noetic-map-server \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-joy \
    ros-noetic-interactive-markers \
    ros-noetic-xacro \
    ros-noetic-rviz \
    ros-noetic-robot-state-publisher \
    python-netifaces \
    python3-netifaces


# # python dependencies
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.9-distutils
RUN apt-get install -y python3.9
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.9 2
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.8 1
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.9 /usr/bin/python3

RUN pip3 install --target /opt/ros/noetic/lib/python3/dist-packages/ netifaces

# Install updated cmake
RUN apt-get update \
    && apt-get -y install build-essential \
    && apt-get install -y wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.24.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

# Make a ROS workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/noetic/setup.sh && catkin_init_workspace src

# Clone the vicon bridge
WORKDIR /catkin_ws/src
RUN git clone https://github.com/ethz-asl/vicon_bridge.git
# Build
WORKDIR /catkin_ws
RUN rosdep install --from-paths src --ignore-src --rosdistro noetic -y
RUN source /opt/ros/noetic/setup.sh && catkin_make

# Clone f1tenth system
WORKDIR /catkin_ws/src
COPY submodules/f1tenth_system /catkin_ws/src/f1tenth_system

# Ignore the packages that we don't need (these are only needed to run onboard the car)
WORKDIR /catkin_ws/src/f1tenth_system
WORKDIR /catkin_ws/src/f1tenth_system/hokuyo_node
RUN touch CATKIN_IGNORE
WORKDIR /catkin_ws/src/f1tenth_system/joystick_drivers
RUN touch CATKIN_IGNORE
WORKDIR /catkin_ws/src/f1tenth_system/rl_runtime
RUN touch CATKIN_IGNORE
WORKDIR /catkin_ws/src/f1tenth_system/scan2pc
RUN touch CATKIN_IGNORE
WORKDIR /catkin_ws/src/f1tenth_system/serial
RUN touch CATKIN_IGNORE
WORKDIR /catkin_ws/src/f1tenth_system/vesc/vesc_driver
RUN touch CATKIN_IGNORE
WORKDIR /catkin_ws/src/f1tenth_system/vesc/vesc
RUN touch CATKIN_IGNORE

# Build
WORKDIR /catkin_ws
RUN apt-get update && apt-get install -y 
RUN rosdep install --from-paths src --ignore-src --rosdistro noetic -y
RUN source /opt/ros/noetic/setup.sh && catkin_make

# Add lines to the bashrc file that source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc


RUN apt-get --purge autoremove python3-pip -y
RUN apt install python3-pip -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install --upgrade numpy
RUN pip3 install seaborn
# RUN pip3 install pyyaml --force-reinstall
RUN pip3 install "jax[cpu]" 
RUN pip3 install rospkg 
RUN pip3 install transforms3d
RUN pip3 install equinox
RUN pip3 install scipy
# RUN apt-get purge python3-yaml -y
# RUN pip3 uninstall pyyaml -y
RUN pip3 install --ignore-installed flax
RUN pip3 install einops
RUN pip3 install colour
RUN pip3 install optax==0.1.7
# RUN pip3 install --upgrade typing_extensions
RUN pip3 install typing_extensions==4.8.0
RUN pip3 install tqdm
RUN pip3 install tensorflow-probability

RUN export PYTHONPATH="${PYTHONPATH}:/usr/lib/"
RUN pip3 install opencv-python 
RUN pip3 install control
RUN pip3 install jraph
RUN pip3 install jaxproxqp@git+https://github.com/realm-robot-ws/jaxproxqp_realm.git
# RUN pip3 install docstring
# RUN apt install python3-netifaces
# RUN pip3 uninstall matplotlib -y
# RUN pip3 install matplotlib --force-reinstall
RUN pip3 install PySide2
RUN apt install rename

RUN apt remove python3-matplotlib -y
RUN pip3 uninstall matplotlib -y
RUN pip3 install matplotlib --upgrade
COPY realm_gc/bash_sh.sh /catkin_ws/src/bash_sh.sh
RUN chmod +x /catkin_ws/src/bash_sh.sh

RUN echo "alias run_ros='rostopic pub /start_control std_msgs/Empty' " >> /root/.bashrc
RUN echo "alias stop_ros='rostopic pub -1 /stop_control std_msgs/Empty' " >> /root/.bashrc

RUN /catkin_ws/src/bash_sh.sh

RUN export DISPLAY=localhost:0.0

RUN apt-get install ffmpeg

RUN pip3 install typing_extensions --upgrade
RUN pip3 install attrs
RUN pip3 install numpy==1.26.4
# RUN pip3 install tkinter
# WORKDIR /catkin_ws/src
# RUN ./bash_sh.sh
# RUN ./bash_sh.sh
# WORKDIR /usr/lib/python3/dist-packages/Cryptodome/Cipher
# RUN rename 's/38/39/' *.so

# WORKDIR /usr/lib/python3/dist-packages/Cryptodome/Hash
# RUN rename 's/38/39/' *.so

# WORKDIR /usr/lib/python3/dist-packages/Cryptodome/Util
# RUN rename 's/38/39/' *.so

# WORKDIR /usr/lib/python3/dist-packages/Cryptodome/Protocol
# RUN rename 's/38/39/' *.so