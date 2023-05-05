import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from mrcnn.config import Config
import random
import colorsys
from matplotlib import patches,  lines

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)
    
class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "coco"
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

def random_colours(N, bright=True):
    """
    Generate random colours.
    To get visually distinct colours, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colours = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colours)
    return colours

def apply_mask(image, mask, colour, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * colour[c] * 255,
                                  image[:, :, c])
    return image

def bounding_boxes(boxes, colour,ax,i):
    y1, x1, y2, x2 = boxes[i]
    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                        alpha=0.7, linestyle="dashed",
                        edgecolor=colour, facecolor='none')
    ax.add_patch(p)
    return x1,y1,y2,x2


def coords(res,boxes,class_names,captions):   
    #Run detection on rgb images 
    #boxes=res['rois']
    masks=res['masks']
    scores=res['scores']
    class_ids=res['class_ids']

    # Number of instances of objs detected
    N = boxes.shape[0]
    
    y1, x1, y2, x2={},{},{},{} #make dicts so coords for each item can be stored
    mask={}

    for i in range(N):
        if not np.any(boxes[i]):
        # Skip this instance. Has no bbox. 
            y1[i], x1[i], y2[i], x2[i]=0,0,0,0
        else:
            y1[i], x1[i], y2[i], x2[i] = boxes[i]
        #Label
        if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        print(caption)
        #Mask
        mask[i] = masks[:, :, i]
    return x1,x2,y1,y2,captions, label,mask
