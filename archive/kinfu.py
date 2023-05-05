import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import sys
import time

def kinfu_demo(colour_image,depth_image):
    cv.ocl.setUseOpenCL(1)
    #mask_depth = np.zeros(depth_image.shape)
       
    #only show the pixels where the object is detected
    mask_depth = np.zeros(depth_image.shape)
    
    #only show the pixels where the object is detected
    #mask_depth[0:200,0:200]=1

    #depth_image *= np.uint16(mask_depth)
    depth_image = np.uint16(depth_image)

    image = depth_image
    (height, width) = image.shape   

    cv.imshow('input', colour_image)

    size = height, width, 4
    cvt8 = np.zeros(size, dtype=np.uint8)
  

    if kf.update(image):
        #kf.reset()
    #else:
        kf.render(cvt8)
        #result=kf.update(depth_image)
        #print(result)
        cv.imshow('render', cvt8)
    
    cv.pollKey()
cv.waitKey(0)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
# params = cv.kinfu_Params.defaultParams()
params = cv.kinfu_Params.hashTSDFParams(False)

kf = cv.kinfu_KinFu.create(params)

while True:
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    colour_frame = frames.get_color_frame()
    colour_image = np.asanyarray(colour_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    
    if depth_frame:
        st=time.time()
        kinfu_demo(colour_image,depth_image)
        en=time.time()
        print('execution time',en-st)
