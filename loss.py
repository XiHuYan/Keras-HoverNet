import numpy as np
import tensorflow as tf
import keras
from keras import losses
import keras.backend as K

def cce(y_true, y_pred):
	return losses.categorical_crossentropy(y_true, y_pred)

def soft_dice(y_true, y_pred):
	smooth = 1.
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)
	inter = K.sum(y_true * y_pred)
	union = K.sum(y_true) + K.sum(y_pred)
	dice  = (inter+smooth) / (union+smooth)
	return 1-dice

def np_loss(l_a, l_b):
	def _loss(y_true, y_pred):
		cce_loss = cce(y_true, y_pred)
		cce_loss = K.mean(cce_loss)
		dice_loss = soft_dice(y_true[...,0], y_pred[...,0]) + soft_dice(y_true[...,1], y_pred[...,1])
		return l_a*cce_loss + l_b*dice_loss
	return _loss

def mse(y_true, y_pred):
	loss = K.square(y_true-y_pred)
	return K.mean(loss)

def gmse(size):
	def get_sobel_kernel(size):
		# kernel size must be odd: 1, 3, 5, 7 ...
		assert size%2!=0
		h_range = np.arange(-size//2+1, size//2+1)
		v_range = np.arange(-size//2+1, size//2+1)
		h, v = np.meshgrid(h_range, v_range)
		h = h/(h*h+v*v+K.epsilon())
		v = v/(h*h+v*v+K.epsilon())
		return h,v

	def _loss(y_true, y_pred):
		kh, kv = get_sobel_kernel(size) # (5,5)
		kh, kv = np.expand_dims(kh, -1), np.expand_dims(kv, -1)
		kernel = np.concatenate([kh,kv], axis=-1)[...,None]         # (H, W, in_depth, out_depth)
		kernel = K.constant(kernel, dtype=tf.float32)

		gt_mask = K.cast(K.greater_equal(y_true[...,0], K.epsilon()), tf.float32)
		M = K.sum(gt_mask)

		# approximate to h/v grad and sum together
		gt_grad = K.conv2d(y_true, kernel, strides=(1,1), padding='same')  # (N,H,W,1)
		pr_grad = K.conv2d(y_pred, kernel, strides=(1,1), padding='same')  

		gmse_loss = K.square(gt_grad-pr_grad)[...,0] * gt_mask
		return K.sum(gmse_loss)/M

	return _loss

def hv_loss(l_c, l_d):
	def _loss(y_true, y_pred):
		mse_loss = mse(y_true, y_pred)
		gmse_loss = gmse(5)(y_true, y_pred)
		return mse_loss + gmse_loss

	return _loss



