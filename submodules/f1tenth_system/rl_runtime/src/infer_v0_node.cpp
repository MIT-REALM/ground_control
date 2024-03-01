#include <ros/ros.h>

#include "infer_v0.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    ros::init(argc, argv, "scan2pc_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh{"~"};

    rl_runtime::InferV0 node{nh, private_nh};

    ros::spin();

    return 0;
}