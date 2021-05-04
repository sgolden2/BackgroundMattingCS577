import numpy as np
from utils import *


def combiner(img,pha,bgr):
	while True:
		x = np.concatenate([next(img),next(bgr)], axis=3).astype(np.uint8)
		y = np.dot(next(pha)[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
		y = np.where(y > (256//2), 1, 0)
		#y = np.reshape(y, y.shape[0:3])
		yield (x,y)

