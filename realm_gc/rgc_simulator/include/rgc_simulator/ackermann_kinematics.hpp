#pragma once

#include "rgc_simulator/Pose2d.h"

namespace racecar_simulator {

class AckermannKinematics {

public:

    static double angular_velocity(
            double velocity,
            double steering_angle,
            double wheelbase);

    static rgc_simulator::Pose2d update(
            const rgc_simulator::Pose2d start,
            double velocity,
            double steering_angle,
            double wheelbase,
            double dt);

};

}
