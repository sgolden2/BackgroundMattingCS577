import numpy as np
import cv2
import random
import glob

# Blending two imgages
def overlay_image_alpha(bgr, fgr, alpha):
    alpha_inv = (1.0 - alpha)
    final = alpha_inv * bgr + alpha * fgr
    return final

bgr = cv2.imread("BACKGROUND")
pha = cv2.imread("ALPHA")/255.0
fgr = cv2.imread("FOREGROUND")
# Resizing the background to the foreground image
bgr = cv2.resize(bgr,(pha.shape[1],pha.shape[0]))

overlay = overlay_image_alpha(bgr,fgr,pha)
cv2.imwrite("overlay.jpg",overlay)