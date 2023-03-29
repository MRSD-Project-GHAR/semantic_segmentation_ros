#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from segmentation_mmseg import *

pub = rospy.Publisher('video_frames_seg', Image, queue_size=10)

def image_callback(data):
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    
    seg_frame = segmentation_mmseg(frame)
    pub.publish(bridge.cv2_to_imgmsg(seg_frame, encoding="rgb8"))
    

def grayscale_subscriber():
    rospy.init_node('grayscale_subscriber', anonymous=True)
    # rospy.Subscriber('video_frames', Image, image_callback)
    rospy.Subscriber('/device_0/sensor_1/Color_0/image/data', Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        grayscale_subscriber()
    except rospy.ROSInterruptException:
        pass

