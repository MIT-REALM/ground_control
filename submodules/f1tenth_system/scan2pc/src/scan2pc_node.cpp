#include <ros/ros.h>

#include "scan2pc.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "scan2pc_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh{"~"};

    scan2pc::Scan2PC scan2pc{nh, private_nh};

    ros::spin();

    return 0;
}
