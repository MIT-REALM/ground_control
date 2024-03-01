#include "scan2pc.h"

#include <sensor_msgs/PointCloud2.h>

namespace scan2pc {
Scan2PC::Scan2PC(ros::NodeHandle nh, const ros::NodeHandle& private_nh) {
    pc_pub_ = nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 1);
    const auto nodelay = ros::TransportHints{}.tcpNoDelay(true);
    scan_sub_ = nh.subscribe("scan", 1, &Scan2PC::scanCb, this, nodelay);
}

void Scan2PC::scanCb(const sensor_msgs::LaserScanConstPtr& msg) {
    // Convert from scan to pointcloud.
    auto pc = boost::make_shared<sensor_msgs::PointCloud2>();
    projector_.projectLaser(*msg, *pc);

    // Publish pointcloud.
    pc_pub_.publish(pc);
}

}  // namespace scan2pc