import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import tensorflow as tf
import cv2
import numpy as np
import ipdb
import pyrealsense2 as rs
import math
import time
import cv2

from mrcnn import model as modellib #, utils
from functions import InferenceConfig,coords, AppState

#setup depth
state = AppState()

def mouse_cb(event, x, y, flags, param): #allow dragging around with mouse

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj

def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation

def pointcloud(out, verts, texcoords, colour, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = colour.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = colour[u[m], v[m]]

def depth(state,depth_frame,colour_frame,colour_image,xstart,xend,ystart,yend):
    if not state.paused:
        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        depth_image = np.asanyarray(depth_frame.get_data())

        mask_depth = np.zeros(depth_image.shape)
       
        #only show the pixels where the object is detected
        mask_depth[xstart:xend,ystart:yend]=1
                
        depth_image *= np.uint16(mask_depth)

        mapped_frame, colour_source = colour_frame, colour_image

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # Render
    now = time.time()

    out.fill(0)
    if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, colour_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, colour_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)


    return mapped_frame,points,depth_image

def kinfu(kf,depth_image):
    cv2.ocl.setUseOpenCL(1) #lets the gpu take the load using opencl

    #mask_depth = np.zeros(depth_image.shape)
    
    #only show the pixels where the object is detected
    #mask_depth[xstart:xend,ystart:yend]=1

    #depth_image = np.uint16(depth_image)

    image = depth_image
    (height, width) = image.shape   

    size = height, width, 4
    cvt8 = np.zeros(size, dtype=np.uint8)
  

    if kf.update(image):
        #kf.reset()
        kf.render(cvt8)
        #result=kf.update(depth_image)
        #print(result)
        cv2.imshow('render', cvt8)

def depth_new(state,depth_frame,depth_image,colour_frame,colour_image,prevmask,xstart,xend,ystart,yend):
    if not state.paused:
        depth_frame = decimate.process(depth_frame)

        # Grab new intrinsics (may be changed by decimation)
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        w, h = depth_intrinsics.width, depth_intrinsics.height

        mask_depth = np.zeros(depth_image.shape)
        mask_depth[xstart:xend,ystart:yend]=1
       
        for x in range(xstart,xend):
            for y in range(ystart,yend):
                mask_depth[x:y]=1
                
        masked_img=prevmask + mask_depth
        #only show the pixels where the object is detected

        depth_image *= np.uint16(masked_img)
        
        mapped_frame, colour_source = colour_frame, colour_image

        points = pc.calculate(depth_frame)
        pc.map_to(mapped_frame)

        # Pointcloud data to arrays
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

    # Render
    now = time.time()

    out.fill(0)
    if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, colour_source)
    else:
        tmp = np.zeros((h, w, 3), dtype=np.uint8)
        pointcloud(tmp, verts, texcoords, colour_source)
        tmp = cv2.resize(
            tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        np.putmask(out, tmp > 0, tmp)

    dt = time.time() - now

    cv2.setWindowTitle(
        state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
        (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

    cv2.imshow(state.WIN_NAME, out)
    key = cv2.waitKey(1)


    return mapped_frame,points,mask_depth

def kinfu_new(kf,depth_mask,depth_image):
    cv2.ocl.setUseOpenCL(1) #lets the gpu take the load using opencl

    # mask_depth = np.zeros(depth_image.shape)
    
    # #only show the pixels where the object is detected
    # mask_depth[xstart:xend,ystart:yend]=1

    # depth_image *= np.uint16(mask_depth)

    depth_image*=np.uint16(depth_mask)

    (height, width) = depth_image.shape
    size =height, width, 4

    cvt8 = np.zeros(size, dtype=np.uint8)
    
    if kf.update(depth_image):
        kf.render(cvt8)
        cv2.imshow('render', cvt8)

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Path to trained weights file
MODEL_PATH="model/mask_rcnn_coco.h5"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

config = InferenceConfig()

# COCO Class names
class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane','bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird','cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie','suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard','surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple','sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed','dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster','sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
               
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_PATH, config=config)

# Load weights trained on MS-COCO
model.load_weights(MODEL_PATH, by_name=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colourizer = rs.colorizer()

cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)

out = np.empty((h, w, 3), dtype=np.uint8)

# params = cv2.kinfu_Params.defaultParams()
params = cv2.kinfu_Params.hashTSDFParams(False)
kf = cv2.kinfu_KinFu.create(params)

captions=None
label=None
try:
    while True:
        if state.paused: #only do this iteration of the loop if state isnt paused
            continue

        label=None
        frames = pipeline.wait_for_frames()

        # Wait for a coherent pair of frames: depth and color
        depth_frame = frames.get_depth_frame()
        colour_frame = frames.get_color_frame()
        colour_image = np.asanyarray(colour_frame.get_data())

        if not depth_frame or not colour_frame:
            continue      

        cv2.imshow('RGB', colour_image)
        
        
        #segmentation
        results=model.detect([colour_image])
        res = results[0]
        boxes=res['rois']
        instances=boxes.shape[0]
        

        xstart,xend,ystart,yend,captions,label,mask=coords(
            res,boxes,class_names,captions)
        
        end=time.time()
        
        depth_image = np.asanyarray(depth_frame.get_data())
        prevmask=np.zeros(depth_image.shape)

        if boxes.any():
            for i in range(instances):
                mapped_frame,points,depth_mask=depth_new(
                state,depth_frame,depth_image,colour_frame,colour_image,prevmask,
                xstart[i],xend[i],ystart[i],yend[i])

            #kinfu(kf,depth_image)
                st=time.time()
                kinfu_new(kf,depth_mask,depth_image)

            end=time.time()
        #print('execution time',end-st)


        
     

finally:
    # Stop streaming
    pipeline.stop()
    # Destroy all the windows
    cv2.destroyAllWindows()
  