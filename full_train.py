from model2 import *
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from datetime import datetime
from utils import *


IMAGE_DIMS = (768,432)
TENSOR_SHAPE = (768,432,6)
BATCH_SIZE = 12
STEPS_PER_EPOCH = 4000
VALIDATION_STEPS = 32
EPOCHS = 10
TRAIN_SUBSET = 0.7



train_img_data = ImageDataGenerator(validation_split=TRAIN_SUBSET)
train_img_gen = train_img_data.flow_from_directory(
	os.path.normpath("./data/train/train_img"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	subset='training',
	batch_size=BATCH_SIZE)

train_pha_data = ImageDataGenerator(validation_split=TRAIN_SUBSET)
train_pha_gen = train_pha_data.flow_from_directory(
	os.path.normpath("./data/train/train_pha"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	subset='training',
	batch_size=BATCH_SIZE)

train_bgr_data = ImageDataGenerator(validation_split=TRAIN_SUBSET)
train_bgr_gen = train_pha_data.flow_from_directory(
	os.path.normpath("./background/tr"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	subset='training',
	batch_size=BATCH_SIZE)

valid_img_data = ImageDataGenerator()
valid_img_gen = valid_img_data.flow_from_directory(
	os.path.normpath("./data/validation/validation_img"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	batch_size=BATCH_SIZE)

valid_pha_data = ImageDataGenerator()
valid_pha_gen = valid_pha_data.flow_from_directory(
	os.path.normpath("./data/validation/validation_pha"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	batch_size=BATCH_SIZE)

valid_bgr_data = ImageDataGenerator()
valid_bgr_gen = valid_pha_data.flow_from_directory(
	os.path.normpath("./background/va"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	batch_size=BATCH_SIZE)


train_datagen = combiner(train_img_gen, train_pha_gen, train_bgr_gen)
valid_datagen = combiner(valid_img_gen, valid_pha_gen, valid_bgr_gen)

K.clear_session()
model = get_full_model(input_shape=TENSOR_SHAPE)
model.summary()
model.fit(train_datagen, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=valid_datagen, validation_steps=VALIDATION_STEPS)

model.save_weights(datetime.now().strftime("%H%M%S_%d%m%Y") + "_FULL_MODEL_WEIGHTS.h5")
