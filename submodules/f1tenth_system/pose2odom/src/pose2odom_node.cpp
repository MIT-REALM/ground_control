#include <ros/ros.h>

#include "pose2odom.h"

int main(int argc, char **argv) {
    ros::init(argc, argv, "pose2odom_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh{"~"};

    pose2odom::Pose2Odom pose2odom{nh, private_nh};

    ros::spin();

    return 0;
}
