uint8 TYPE_RPM=0
uint8 TYPE_DUTY=1
uint8 TYPE_CURRENT=2
uint8 TYPE_CURRENT_BRAKE=3

# Which command to apply.
uint8 type

# Revolutions per minute.
float64 rpm
# Duty cycle, [0, 1].
float64 duty
# Current A.
float64 current
# Current brake.
float64 current_brake
# Servo position.
float64 servo_position