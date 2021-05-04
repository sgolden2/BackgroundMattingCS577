from model2 import *
import tensorflow as tf
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np



IMAGE_DIMS = (768,432)
TENSOR_SHAPE = (768,432,6)
BATCH_SIZE = 8
STEPS_PER_EPOCH = 16


def alpha_mse_loss(pha_true, base_out):
	pha_pred,_,_,_ = base_out
	return tf.square(pha_true - pha_pred)


train_img_data = ImageDataGenerator(validation_split=0.999)
train_img_gen = train_img_data.flow_from_directory(
	os.path.normpath("./data/train/train_img"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	subset='training',
	batch_size=BATCH_SIZE)

train_pha_data = ImageDataGenerator(validation_split=0.999)
train_pha_gen = train_pha_data.flow_from_directory(
	os.path.normpath("./data/train/train_pha"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	subset='training',
	batch_size=BATCH_SIZE)

train_bgr_data = ImageDataGenerator(validation_split=0.999)
train_bgr_gen = train_pha_data.flow_from_directory(
	os.path.normpath("./background/tr"), 
	target_size=IMAGE_DIMS,
	class_mode=None,
	subset='training',
	batch_size=BATCH_SIZE)

'''
valid_img_data = ImageDataGenerator(rescale=1./255)
valid_img_gen = valid_img_data.flow_from_directory(
	os.path.normpath("./data/validation/img"), 
	target_size=IMAGE_DIMS, 
	batch_size=BATCH_SIZE)

valid_pha_data = ImageDataGenerator(rescale=1./255)
valid_pha_gen = valid_pha_data.flow_from_directory(
	os.path.normpath("./data/validation/pha"), 
	target_size=IMAGE_DIMS, 
	batch_size=BATCH_SIZE)

'''
def combiner(img,pha,bgr):
	while True:
		x = np.concatenate([next(img),next(bgr)], axis=3).astype(np.uint8)
		y = np.dot(next(pha)[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
		y = np.where(y > (256//2), 1, 0)
		#y = np.reshape(y, y.shape[0:3])
		yield (x,y)


datagen = combiner(train_img_gen, train_pha_gen, train_bgr_gen)


model = get_base_model(input_shape=TENSOR_SHAPE)
model.summary()
model.fit(datagen, steps_per_epoch=STEPS_PER_EPOCH)
