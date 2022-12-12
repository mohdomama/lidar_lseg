# ROS Communication Utilities
import rospy
import sensor_msgs
import std_msgs
from sensor_msgs.msg import PointCloud2, PointField, Joy
from std_msgs.msg import Header
from nav_msgs.msg import Path, Odometry
from roslib import message
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped
# from util.transforms import get_rpy_from_odom_orientation
import ros_numpy


import numpy as np
import ros_numpy
import json
from pyloam.src.feature_extract import FeatureExtract



class RosCom:
    def __init__(self) -> None:

        # rospy.Subscriber('/aft_mapped_to_init_high_frec',
        #                  Odometry, self.loam_odom_callback)
        # rospy.Subscriber('/laser_odom_to_init', Odometry,
        #                  self.loam_odom_nomap_callback)
        rospy.Subscriber('/aft_mapped_to_init', Odometry,
                         self.odom_aftmap_callback)

        rospy.Subscriber('/laser_cloud_map', PointCloud2,
                         self.map_callback)

        self.points_publisher = rospy.Publisher(
            '/velodyne_points', PointCloud2, queue_size=1)

        self.loam_latest = [0, 0, 0, 0]
        self.loam_map_latest = [0, 0, 0, 0]

        self.msg = Path()
        self.msg.header.frame_id = 'camera_init'
        self.msg.header.stamp = rospy.Time.now()
        self.pcd = None

        self.old_map = None



    # def loam_odom_nomap_callback(self, msg):
    #     # just odometry
    #     self.odom_nomap_seq = msg.header.seq
    #     position = msg.pose.pose.position
    #     orientation = msg.pose.pose.orientation
    #     roll, pitch, yaw = get_rpy_from_odom_orientation(
    #         [orientation.x, orientation.y, orientation.z, orientation.w])
    #     self.loam_nomap_latest = np.array([position.x, position.y, position.z, roll, pitch, yaw])

    def odom_aftmap_callback(self,msg):
        self.odom_nomap_seq = msg.header.seq
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        print("Yo Odom!")




    def map_callback(self, msg):
        pc = ros_numpy.numpify(msg)
        print(type(pc))
        # print(pc.shape)

    def clear(self):
        self.sequences = []
        self.positions = []

    def pcd_2_point_cloud(self, points, parent_frame, frametime):
        assert points.shape[1] == 5, 'PCD should be in XYZIR format!'
        ros_dtype = sensor_msgs.msg.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize
        data = points.astype(dtype).tobytes()
        fields = [
            sensor_msgs.msg.PointField(
                name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate(['x', 'y', 'z', 'intensity', 'ring'])
        ]
        header = std_msgs.msg.Header(frame_id=parent_frame, stamp=frametime)

        return sensor_msgs.msg.PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 5),
            row_step=(itemsize * 5 * points.shape[0]),
            data=data
        )

    def publish_points(self, pcd):
        assert pcd.shape[1] == 3, 'PCD should be in XYZ format'
        # Add  intensity and ring channel
        pcd = np.hstack([pcd, np.ones((pcd.shape[0], 1))])
        pcd, scan_id = self.feature_extract.get_scan_id(pcd)
        pcd = np.hstack((pcd, scan_id.astype(np.float32)))

        ros_pcd = self.pcd_2_point_cloud(pcd, 'map', rospy.Time.now())
        self.points_publisher.publish(ros_pcd)



def main():
    rospy.init_node('rosutil')
    roscom = RosCom()
    rospy.spin()


if __name__ == '__main__':
    main()
