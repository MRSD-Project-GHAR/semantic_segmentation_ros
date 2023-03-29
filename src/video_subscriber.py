#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from segmentation_mmseg import *

# Access the rosparams
param_subcriber_topic = rospy.get_param('subcriber_topic')
param_publisher_topic = rospy.get_param('publisher_topic')

pub = rospy.Publisher(param_publisher_topic, Image, queue_size=10)

def image_callback(data):
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    
    seg_frame = segmentation_mmseg(frame)
    pub.publish(bridge.cv2_to_imgmsg(seg_frame, encoding="rgb8"))
    

def image_subscriber():
    rospy.init_node('image_subscriber', anonymous=True)
    rospy.Subscriber(param_subcriber_topic, Image, image_callback)

    rospy.spin()

if __name__ == '__main__':
    try:
        image_subscriber()
    except rospy.ROSInterruptException:
        pass

