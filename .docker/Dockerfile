FROM osrf/ros:noetic-desktop-full

# Install packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ros-noetic-driver-base

# Make a ROS workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws
SHELL ["/bin/bash", "-c"]  # so we can use `source`
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
RUN git clone https://github.com/MIT-REALM/f1tenth_system.git
# Build
WORKDIR /catkin_ws
RUN rosdep install --from-paths src --ignore-src --rosdistro noetic -y
RUN source /opt/ros/noetic/setup.sh && catkin_make

# Add lines to the bashrc file that source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc
