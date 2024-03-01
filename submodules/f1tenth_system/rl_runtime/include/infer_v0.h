#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

#include <filesystem>

#include "trt_model.h"

namespace rl_runtime {
namespace fs = std::filesystem;

class InferV0 {
   public:
    InferV0(ros::NodeHandle nh, const ros::NodeHandle& private_nh);

   private:
    float servo_pos_offset_;
    float servo_pos_gain_;

    float v_coeff_;

    /// Controller frequency.
    double dt_min_;

    /// Initial lidar angle.
    double obs_lidar_theta_i_;
    /// Final lidar angle.
    double obs_lidar_theta_f_;
    /// Number of lidar points in the observations.
    int obs_n_lidar_;

    float max_hit_dist_;

    uint32_t cmd_seq_;

    void ScanCb(const sensor_msgs::LaserScanConstPtr& msg);
    void OdomCb(const nav_msgs::OdometryConstPtr& msg);
    void NavGoalCb(const geometry_msgs::PoseStampedConstPtr& msg);

    void TryRunAndPublish();
    void RunAndPublish();

    [[nodiscard]] float CurvatureToServoPos(float kappa) const;

    std::unique_ptr<TRTModel> trt_model_;

    ros::Time last_run_stamp_;

    sensor_msgs::LaserScanConstPtr scan_;
    nav_msgs::OdometryConstPtr odom_;
    geometry_msgs::PoseStampedConstPtr nav_goal_;

    sensor_msgs::LaserScan scan_dbg_;

    ros::Subscriber odom_sub_;
    ros::Subscriber scan_sub_;
    ros::Subscriber nav_goal_sub_;
    ros::Publisher cmd_pub_;
    ros::Publisher lidar_pub_;
};
}  // namespace rl_runtime