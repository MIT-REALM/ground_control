FROM osrf/ros:noetic-desktop-full

# Download and build the required franka libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git

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

# Add lines to the bashrc file that source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "source /catkin_ws/devel/setup.bash" >> /root/.bashrc
