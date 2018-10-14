from pykinect2 import PyKinectRuntime
from pykinect2 import PyKinectV2
import cv2
import numpy as np
import pickle
import os
import sys

kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectV2.FrameSourceTypes_Color |
    PyKinectV2.FrameSourceTypes_Depth)

while True:
    if kinect.has_new_color_frame() and \
            kinect.has_new_depth_frame():
        ##get rgb frame
        rgba_frame = kinect.get_last_color_frame()
        rgba_frame = rgba_frame.reshape((
                                        kinect.color_frame_desc.Height,
                                        kinect.color_frame_desc.Width,
                                        4), order='C')
        rgb_frame = cv2.cvtColor(rgba_frame,
                                 cv2.COLOR_RGBA2RGB)

        # get depth frame
        depth_frame = kinect.get_last_depth_frame()
        f8 = np.uint8(depth_frame.clip(0, 5100) / 20.)
        framecv2 = f8.reshape((424, 512))

        cv2.imshow('kinectRGB', rgb_frame.astype('uint8'))
        cv2.imshow('kinectDepth', framecv2)

        ##save file
    if cv2.waitKey(1) == 27:
        if not os.path.exists('CameraData'):
            os.makedirs('CameraData')
        file_name = sys.argv[1] + '.p'
        pickle_path = os.path.join('CameraData', file_name)
        pickling_on = open(pickle_path, "wb")
        pickle.dump(dict(
            [('rgb', rgb_frame), ('depth', depth_frame)]),
                    pickling_on)
        pickling_on.close()
        break

cv2.destroyAllWindows()
