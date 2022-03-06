#!/usr/bin/env python

import rospy
import numpy as np

from visual_servoing.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped

class ParkingController():
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """
    W_B = 0.325 # Wheel Base
    G_E = 0.3 # good enough distance
    SPEED_CONST = 0.2
    DEG_THRESH = 2 * np.pi/180

    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation,
            self.relative_cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic") 
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC,
            AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",
            ParkingError, queue_size=10)
        self.parking_distance = 0.75
        self.relative_x = 0
        self.relative_y = 0

    def relative_cone_callback(self, msg):
        x_pos = msg.x_pos
        y_pos = msg.y_pos
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = self.dr_vel(x_pos,y_pos)
        drive_cmd.drive.steering_angle = self.dr_ang(x_pos,y_pos)
        self.error_publisher(x_pos,y_pos)
        self.drive_pub.publish(drive_cmd)

    def dr_ang(self,x,y):
        infront = x > 0
        indeg = np.arctan(x/self.parking_distance) < self.DEG_THRESH
        if infront and indeg:
            output = 0
        else:
            cr = self.parking_distance
            tr = ((y**2)+(x**2)-(cr**2))/(2*y)
            ang = np.arctan(self.W_B/tr)
            output = max(min(ang,0.34),-0.34)
        return output

    def dr_vel(self,x,y):
        d = np.linalg.norm((x,y)) - self.parking_distance
        infront = x > 0
        inrange = abs(d) <= self.G_E
        if inrange and infront:
            output = 0
        else:
            output = d #(self.SPEED_CONST*d)**3
        return output

    def error_publisher(self,x,y):
        error_msg = ParkingError()
        error_msg.x_error = x - self.parking_distance
        error_msg.y_error = y
        error_msg.distance_error = np.linalg.norm((x,y)) - self.parking_distance
        self.error_pub.publish(error_msg)

if __name__ == '__main__':
    try:
        rospy.init_node('ParkingController', anonymous=True)
        ParkingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
