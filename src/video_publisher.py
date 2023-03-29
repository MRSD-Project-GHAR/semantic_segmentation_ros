#!/usr/bin/env python3
import rospy, rospkg
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rospkg = rospkg.RosPack()
pkg_path = rospkg.get_path('semantic_segmentation_ros')
video_path = pkg_path + '/media/video.mp4'

def video_publisher(video_path=None):
    
    rospy.init_node('video_publisher', anonymous=True)
    pub = rospy.Publisher('video_frames', Image, queue_size=10)
    bridge = CvBridge()
    
    if not video_path:
        video_capture = cv2.VideoCapture(0)
    else:
        video_capture = cv2.VideoCapture(video_path)
    rate = rospy.Rate(30)  # 30 Hz to match common video frame rates

    while not rospy.is_shutdown() and video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            image_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            pub.publish(image_msg)
        else:
            break
        rate.sleep()

    video_capture.release()

if __name__ == '__main__':
    try:
        video_publisher()
    except rospy.ROSInterruptException:
        pass

