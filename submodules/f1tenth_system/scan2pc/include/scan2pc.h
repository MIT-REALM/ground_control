#pragma once

#include <laser_geometry/laser_geometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

namespace scan2pc {
class Scan2PC {
   public:
    Scan2PC(ros::NodeHandle nh, const ros::NodeHandle& private_nh);

   private:
    void scanCb(const sensor_msgs::LaserScanConstPtr& msg);

    laser_geometry::LaserProjection projector_;

    ros::Subscriber scan_sub_;
    ros::Publisher pc_pub_;
};
}  // namespace scan2pc
