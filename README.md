# REALM Ground Control

This repository provides infrastructure for operating the REALM ground robots (F1tenth & Turtlebots). It is currently based on ROS 1 Noetic (for compatibility with the F1tenth and Turtlebots).

## Installation

This repository is currently only supported on Ubuntu Linux, but it may be possible to use with other operating systems (if you do this, feel free to submit a PR with instructions).

1. Install Docker by following the instructions [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). Install using the `apt` repository rather than using the Docker Desktop tool.
2. Follow the post-installation steps for Docker [here](https://docs.docker.com/engine/install/linux-postinstall/). If you don't follow these steps, you will only be able to run Docker using `sudo` and will have many headaches as a result. Note: you may need to restart your system rather than just logging out and logging back in.
3. Verify that Docker has been installed correctly by running `docker run hello-world` from a terminal. This should print a message and then exit.
4. Clone this repository

```bash
git clone --recurse-submodules git@github.com:MIT-REALM/ground_control.git
cd ground_control
```

5. Build the docker images

```bash
docker compose build
```

## Highbay Setup

_These instructions are tailored for the East High Bay in Building 31 at MIT._

1. Make sure your computer is connected to the same network as the Vicon computer (via ethernet into the router or by connection to the RAVEN-50 WiFI network using the password written on the router)._
2. Make sure the Vicon computer is on (it should NEVER be turned off).
3. Turn on the Vicon cameras by turning on the extension cable on the shelf by the door between the two control rooms.
4. Make sure whatever objects you want to track are registered and enabled in the Vicon software on the Vicon computer.

After following these steps, you should be ready to use the ROS interface.

This system includes a `docker-compose.yml` file that defines 4 services, which will each run in a different Docker container:

- `vicon_bridge`: runs the ROS node that interfaces with the Vicon system.
- `roscore`: runs the ROS core instance.
- `bash`: provides a bash shell where you can run commands like `rostopic echo ...`
- `rviz`: launches RViz to visualize the ROS environment

If you want to run all of these services, simply run `docker compose up`. If you want to connect to the `bash` service, run `docker attach highbay-1-bash` in another terminal tab, and to disconnect type `Ctrl+p` then `Ctrl-q`. To stop, use `Ctrl+C` in the terminal where you ran `docker compose up` (you may need to manually close the RViz window).

If you only want to run the `vicon_bridge` service (e.g. if you want to start your own ROS core using `roslaunch`), then just run `docker compose start vicon_bridge` after you have started `roscore`. To stop it, run `docker compose stop vicon_bridge`.

## Robot setup

TODO document how to set ROS_MASTER_URI and ROS_HOSTNAME on all robots

- Get the IP address of the host laptop on the wifi network you'll be using (`RAVEN-50` in the east high bay) by running `ip addr` and getting the IP address listed under `wlan0`.
- Set the `ROS_MASTER_URI` and `ROS_HOSTNAME` environment variables on all robots.
    - Get the IP address of the robot by running `ip addr` while ssh'd into the robot.
    - Edit `~/.bashrc` with `export ROS_MASTER_URI=http://<host laptop ip>:11311` and `export ROS_HOSTNAME=<ip of robot>`.
    - Run `source ~/.bashrc` before doing anything else.

Whenever you start up the F1Tenth, make sure it connects to the right wifi (currently, you need to hook it up to a monitor/keyboard/mouse and manually connect). Also, when you ssh into the F1Tenth, make sure you `cd ~/sandbox/f1tenth_ws && source devel/setup.bash`

## Running experiments

After putting all robots in their start positions:

1. **Start ROS core and main stack:** Run `docker compose up` to start all services
2. **Start TB1:** SSH into turtle 1 and run `ROS_NAMESPACE=turtle1 roslaunch turtlebot3_bringup turtlebot3_robot.launch`
3. **Start TB2:** SSH into turtle 2 and run `ROS_NAMESPACE=turtle2 roslaunch turtlebot3_bringup turtlebot3_robot.launch`
4. **Start F1Tenth:** SSH into f1tenth and run `roslaunch racecar teleop.launch`
5. **Start camera:** SSH into f1tenth again and run `roslaunch realsense2_camera rs_camera.launch depth_width:=256 depth_height:=144 depth_fps:=90 enable_color:=false`
5. **Start experiments:** Attach to the bash service `docker attach bash`
    a. Enable teleop control of F1Tenth by holding R1 on the controller
    b. Start the experiment by `rostopic pub -1 /start_control`
    c. Stop the experiment by `rostopic pub -1 /stop_control`

## TODO

Configure environment variables
```
ENV ROS_MASTER_URI=http://localhost:11311
ENV ROS_IP=192.168.0.166
```