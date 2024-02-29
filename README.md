# Vicon-ROS Bridge Docker

This repository provides Docker images that interface with a Vicon motion capture system. It also provides a `docker compose` system for launching a complete ROS environment that connects to the Vicon system.

## Installation

This repository is currently only supported on Ubuntu Linux, but it may be possible to use with other operating systems (if you do this, feel free to submit a PR with instructions).

1. Install Docker by following the instructions [here](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository). Install using the `apt` repository rather than using the Docker Desktop tool.
2. Follow the post-installation steps for Docker [here](https://docs.docker.com/engine/install/linux-postinstall/). If you don't follow these steps, you will only be able to run Docker using `sudo` and will have many headaches as a result. Note: you may need to restart your system rather than just logging out and logging back in.
3. Verify that Docker has been installed correctly by running `docker run hello-world` from a terminal. This should print a message and then exit.
4. Clone this repository

```bash
git clone git@github.com:MIT-REALM/highbay.git
cd highbay
```

5. Build the docker images

```bash
docker compose build
```

## Setup

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