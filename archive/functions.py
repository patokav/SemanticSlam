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
    print(np.where(mask==1)[0])
    print(np.where(mask==1)[0][0])
    print(np.where(mask==1)[0][-1])
    print(np.where(mask==1)[1])
    print(np.where(mask==1)[1][0])
    print(np.where(mask==1)[1][-1])
    return image

def bounding_boxes(boxes, colour,ax,i):
    y1, x1, y2, x2 = boxes[i]
    p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                        alpha=0.7, linestyle="dashed",
                        edgecolor=colour, facecolor='none')
    ax.add_patch(p)
    return x1,y1,y2,x2


def coords(frame,model,class_names,captions):   
    #Run detection on rgb images 
    results=model.detect([frame])
    res = results[0]

    boxes=res['rois']
    masks=res['masks']
    scores=res['scores']
    class_ids=res['class_ids']
    # Number of instances of objs detected

    N = boxes.shape[0]
    print("boxes.shape 0 ", N)
    # #FIND WAY TO PASS THROUGH COORDS FOR ALL OBJS
    # if not N:
    #     print("\n*** No instances to display *** \n")
    # else:
    #     assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]   

    for i in N:
        if not np.any(boxes[i]):
        # Skip this instance. Has no bbox. 
            y1, x1, y2, x2=0,0,0,0
        else:
            y1, x1, y2, x2 = boxes[i]
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
        mask = masks[:, :, i]
    return x1,x2,y1,y2,captions, label
    #return np.where(mask==1),captions #get coords of obj

def mouse_cb(event, x, y, flags, param,state,out): #allow dragging around with mouse

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


