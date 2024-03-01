#include "logger_trt.h"
#include "fmt/format.h"
//#include "ros/ros.h"

namespace rl_runtime {
void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
//    switch (severity) {
//        case Severity::kINTERNAL_ERROR:
//            ROS_ERROR(msg);
//            break;
//        case Severity::kERROR:
//            ROS_ERROR(msg);
//            break;
//        case Severity::kWARNING:
//            ROS_WARN(msg);
//            break;
//        case Severity::kINFO:
//            ROS_INFO(msg);
//            break;
//        case Severity::kVERBOSE:
//            ROS_DEBUG(msg);
//            break;
//    }
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
            fmt::print("INTERNAL_ERROR: {}\n", msg);
            break;
        case Severity::kERROR:
            fmt::print("ERROR: {}\n", msg);
            break;
        case Severity::kWARNING:
            fmt::print("WARN: {}\n", msg);
            break;
        case Severity::kINFO:
            fmt::print("INFO: {}\n", msg);
            break;
        case Severity::kVERBOSE:
            fmt::print("DEBUG: {}\n", msg);
            break;
    }
}
}  // namespace rl_runtime
