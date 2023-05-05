import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import tensorflow as tf
import cv2
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import ipdb
import os
import open3d as o3d
import tensorflow as tf
import cv2
import numpy as np
import ipdb
import pyrealsense2 as rs
import math
import time

from mrcnn import model as modellib, utils
from functions import random_colours, apply_mask, InferenceConfig,AppState

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Path to trained weights file
MODEL_PATH="model/mask_rcnn_coco.h5"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

config = InferenceConfig()
#config.display()

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

state=AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color,  640,360, rs.format.bgr8, 30)

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

out = np.empty((h, w, 3), dtype=np.uint8)


#set image or live video segmentation
mode=True #set True for live vid, set false for image and set img variable to desired image
img='test.jpeg'
if mode == True:
    #Load live video feed
    #cam=cv2.VideoCapture(1) #vid capture 0 is depth cam

    while True:
        #capture frame by frame
        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        colour_frame = frames.get_color_frame()
        colour_image = np.asanyarray(colour_frame.get_data())

        frame=colour_image
        cv2.imshow('RGB', frame)
        
        #Run detection on rgb images 
        results=model.detect([frame])
        res = results[0]

        boxes=res['rois']
        masks=res['masks']
        scores=res['scores']
        class_ids=res['class_ids']
        instances=boxes.shape[0]
        # Number of instances of objs detected

        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        #If no axis is passed, create one and automatically call show()
        auto_show = False
        _, ax = plt.subplots(1, figsize=(10,10))
        #auto_show = True

        #Generate random colours
        colours = random_colours(N)

        # Show area outside image boundaries.
        height, width = frame.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        captions=None
        show_mask=True

        masked_image = frame.astype(np.uint32).copy()
        for i in range(N):
            colour = colours[i]

        #Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. 
            continue
        y1, x1, y2, x2 = boxes[i]
        #if show_bbox:
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                            alpha=0.7, linestyle="dashed",
                            edgecolor=colour, facecolor='none')
        ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
            print(caption)
        ax.text(x1, y1 + 8, caption,
                color='b', size=11, backgroundcolor="none")
        
        print(caption)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, colour)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=colour)
            ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8))
        plt.show()

        #m=cv2.bitwise_and(frame,masked_image.astype(np.uint8))
        #stream=cv2.bitwise_and(m,p)
        
        #cv2.imshow("mask",stream)

        #ipdb.set_trace() #breakpoint

        
        #plt.clf()

        # # press 'q' to quit 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# After the loop release the cap object
#cam.release()
# Destroy all the windows
cv2.destroyAllWindows()