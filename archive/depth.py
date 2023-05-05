import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

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

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue


        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        (height, width) = depth_image.shape
        print("height ", height)
        print("width", width)
        color_image = np.asanyarray(color_frame.get_data())

        mask = np.zeros(depth_image.shape)
        rows, cols = depth_image.shape
        #mask[rows//4:3*rows//4, cols//4:3*cols//4]=1
        #depth_image *= np.uint16(mask)
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_colormap)
        #quit program if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pc = rs.pointcloud()
        pcd = pc.calculate(depth_frame)
        points = (np.asarray(pcd.get_vertices()))
        #o3d.visualization.draw_geometries([pcd.get_vertices()])
        #o3d.visualization.draw_geometries()


finally:

    # Stop streaming
    pipeline.stop()

    # After the loop destroy all the windows
    cv2.destroyAllWindows()