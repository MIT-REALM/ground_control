#!usr/bin/env python
"""Define class for robot control """
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import time

class RobotControl():

    def __init__(self):
        rospy.init_node('robot_control_node',anonymous=True)
        self.vel_publisher = rospy.Publisher('/cmd_vel',Twist,queue_size=1) 

        """To-do: change subscription to 'state estimation' topic. Add trajectory param"""
        self.laser_subscriber = rospy.Subscriber('/kobuki/laser/scan',LaserScan,self.laser_callback) 
        self.summit_laser_subscriber = rospy.Subscriber('/hokuyo_base/scan',LaserScan,self.summit_laser_callback) 
        self.cmd = Twist()
        self.laser_msg = LaserScan()
        self.summit_laser_msg = LaserScan()
        self.ctrl_c = False
        self.rate = rospy.Rate(1) #Loop freqquench Hz
        rospy.on_shutdown(self.shutdownhook)
    def publish_once_in_cmd_vel(self):
        while not self.ctrl_c:
            connections  = self.vel_publisher.get_num_connections()
            if connections>0:
                self.vel_publisher.publish(self.cmd) 
                break
            else:
                self.rate.sleep()

    def shutdownhook(self):
        self.ctrl_c =True

    """Need to rewrite code from down here and include some methods for run, reset, and update"""
    def laser_callback(self,msg):
        self.laser_msg = msg

    def summit_laser_callback(self,msg):
        self.summit_laser_msg = msg
    
    def get_laser(self,pos):
        time.sleep(1)
        return self.laser_msg.ranges[pos]
    
    def get_summit_laser(self,pos):
        time.sleep(1)
        return self.laser_msg.ranges[pos]
    
    def get_laser_full(self):
        time.sleep(1)
        return self.laser_msg.ranges
    
    def stop_robot(self):
        self.cmd.linear.x = 0.0
        self.cmd.angular.z = 0.0
        self.publish_once_in_cmd_vel()
    
    def move_straight(self):
        self.cmd.linear.y = 0.0
        self.cmd.linear.z = 0.0
        self.cmd.angular.x = 0.5
        self.cmd.angular.y = 0.0
        self.cmd.angular.z = 0.0
        self.publish_once_in_cmd_vel()

    def move_straight_time(self,motion,speed,time):
        self.cmd.linear.y = 0.0
        self.cmd.linear.z = 0.0
        self.cmd.angular.x = 0.0
        self.cmd.angular.y = 0.0
        self.cmd.angular.z = 0.0

        if motion=='forward':
            self.cmd.linear.x = speed
        elif motion =='backward':
            self.cmd.linear.x = -speed
        
        i=0

        while (i<=time):

            self.vel_publisher.publish(self.cmd)
            i+=1
            self.rate.sleep()

        self.stop_robot()

if __name__=='__main__':
    robotcontrol_object  = RobotControl()
    try:
        robotcontrol_object.move_straight()
    except rospy.ROSInterruptException:
        pass