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
    DIST_THRESH = 0.1 # good enough distance
    OUT_THRESH = 2 # good enough distance
    ANGL_THRESH = 2*np.pi/180
    SPEED_CONST = 0.2
    

    def __init__(self):
        rospy.Subscriber("/relative_cone", ConeLocation,
            self.relative_cone_callback)

        DRIVE_TOPIC = rospy.get_param("~drive_topic") 
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC,
            AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error",
            ParkingError, queue_size=10)
        self.parking_distance = 3
        self.relative_x = 0
        self.relative_y = 0
        self.rev = False

    def relative_cone_callback(self, msg):
        x_pos = msg.x_pos
        y_pos = msg.y_pos
        drive_cmd = AckermannDriveStamped()
        drive_cmd.drive.speed = self.dr_vel(-y_pos,x_pos)
        drive_cmd.drive.steering_angle = self.dr_ang(-y_pos,x_pos)
        self.error_publisher(x_pos,y_pos)
        self.drive_pub.publish(drive_cmd)

    def dr_ang(self,x,y):
        angl = np.arctan(x/y)
        cr = self.parking_distance
        tr = ((y**2)+(x**2)-(cr**2))/(2*x) if x != 0 else 0
        ang = -np.arctan(self.W_B/tr)
        output = max(min(ang,0.34),-0.34)
        return output

    def dr_vel(self,x,y):
        angl = np.arctan(x/y)
        dist = np.linalg.norm((x,y)) - self.parking_distance
        inangl = abs(angl) <= self.ANGL_THRESH
        indist = abs(dist) <= self.DIST_THRESH
        self.rev = (not self.rev) if (indist and (not inangl)) or ((dist > self.OUT_THRESH) and self.rev) else self.rev

        if self.rev:
            output = -1
        else:
            if indist:
                output = 0
            else:
                output = 1*np.sign(dist)

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
