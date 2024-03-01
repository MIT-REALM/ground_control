## Driving command for a car-like vehicle using Ackermann steering.

# Assumes Ackermann front-wheel steering. The left and right front
# wheels are generally at different angles. To simplify, the commanded
# angle corresponds to the yaw of a virtual wheel located at the
# center of the front axle, like on a tricycle.  Positive yaw is to
# the left. (This is *not* the angle of the steering wheel inside the
# passenger compartment.)
#
# Zero steering angle velocity means change the steering angle as
# quickly as possible. Positive velocity indicates a desired absolute
# rate of change either left or right. The controller tries not to
# exceed this limit in either direction, but sometimes it might.
#
uint8 MODE_STEER_ANGLE = 0
uint8 MODE_SERVO = 1
uint8 steer_mode

float32 steering_angle          # desired virtual angle (radians)
float32 servo_position          # desired servo position.

# Drive at requested speed or acceleration (the 1st or 2nd derivatives
# of position). 
uint8 MODE_SPEED = 0
uint8 MODE_ACCEL = 1
uint8 mode

# Speed is the desired scalar magnitude of the velocity vector.
# Direction is forward unless the sign is negative, indicating reverse.
#
# Zero acceleration means change speed as quickly as
# possible. Positive acceleration indicates a desired absolute
# magnitude; that includes deceleration.
#
float32 speed                   # desired forward speed (m/s)
float32 acceleration            # desired acceleration (m/s^2). Negative indicates brake.
