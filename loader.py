from model2 import *
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from utils import *


TENSOR_SHAPE = (768,432,6)
IMAGE_DIMS = (768,432)
BATCH_SIZE = 8

model = get_base_model(input_shape=TENSOR_SHAPE)
model.load_weights('193717_04052021_MODEL_WEIGHTS.h5')


test_img_data = ImageDataGenerator()
test_img_gen = test_img_data.flow_from_directory(
	os.path.normpath("./data/test/test_img"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	batch_size=BATCH_SIZE)

test_pha_data = ImageDataGenerator()
test_pha_gen = test_pha_data.flow_from_directory(
	os.path.normpath("./data/test/test_pha"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	batch_size=BATCH_SIZE)

test_bgr_data = ImageDataGenerator()
test_bgr_gen = test_pha_data.flow_from_directory(
	os.path.normpath("./background/te"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	batch_size=BATCH_SIZE)


test_datagen = combiner(test_img_gen, test_pha_gen, test_bgr_gen)

preds = model.predict(test_datagen, steps=10)