from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Activation, BatchNormalization
from keras.optimizers import Adam, SGD
import keras.backend as K
import numpy as np
from resnet50 import _relu, _conv_relu, _conv_bn_relu, _res_stage
from keras.layers import UpSampling2D, Cropping2D, Concatenate

def _dense_blk(inputs, n_units, act=_relu, _con_blk=_conv_relu):
	for i in range(n_units):
		conv = act(inputs)
		conv = _con_blk(128, 1, strides=(1,1), padding='valid')(conv)
		conv = Conv2D(32, 5, strides=(1,1), padding='valid')(conv)

		inputs = Cropping2D((2,2))(inputs)
		inputs = Concatenate(axis=-1)([inputs, conv])
	return inputs

def encoder(inputs):
    base_chnl = 64
    stage_chls = 2**np.arange(0,4) * base_chnl # 64, 128, 256, 512
    stage_units = [3,4,6,3]

    # conv1
    conv1 = _conv_relu(base_chnl, 7, strides=(1,1), padding='valid')(inputs)

    # pool1 
    # pool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1)

    # stage 2
    stage1 = _res_stage(stage_chls[0],   stage_chls[0], stage_chls[0]*4, 1, conv1, stage_units[0])
    stage2 = _res_stage(stage_chls[0]*4, stage_chls[1], stage_chls[1]*4, 2, stage1, stage_units[1])
    stage3 = _res_stage(stage_chls[1]*4, stage_chls[2], stage_chls[2]*4, 2, stage2, stage_units[2])
    stage4 = _res_stage(stage_chls[2]*4, stage_chls[3], stage_chls[3]*4, 2, stage3, stage_units[3])

    stage4 = Conv2D(1024, 1, strides=(1,1), padding='same')(stage4)   # 这里不带activation
    return [stage1, stage2, stage3, stage4]

def decoder(encoder):
	chnls = 2**np.arange(1,5) * 64 
	d1, d2, d3, d4 = encoder
	conv1 = UpSampling2D(size=(2,2))(d4)  # 66, 1024
	conv1 = Add()([d3, conv1])
	conv1 = Conv2D(256, 5, strides=(1,1), padding='valid')(conv1)  # 62, 256, 5*5没有actvtion
	dense1 = _dense_blk(conv1, 8, _relu, _conv_relu)
	conv1 = Conv2D(512, 1, strides=(1,1), padding='valid')(dense1) # no activation

	conv2 = UpSampling2D(size=(2,2))(conv1)
	conv2 = Add()([d2, conv2])
	conv2 = Conv2D(128, 5, strides=(1,1), padding='valid')(conv2)  # no activation
	dense2 = _dense_blk(conv2, 4, _relu, _conv_relu)
	conv2 = Conv2D(256, 1, strides=(1,1), padding='valid')(dense2) # no activation

	conv3 = UpSampling2D(size=(2,2))(conv2)
	conv3 = Add()([d1, conv3])
	conv3 = Conv2D(64, 5, strides=(1,1), padding='same')(conv3)    # no activation

	return conv3

def hvnet(input_shape, n_chnl):
	inputs = Input(input_shape+(n_chnl,))
	enc = encoder(inputs)

	enc[0] = Cropping2D((92,92))(enc[0])  # crop features 
	enc[1] = Cropping2D((36,36))(enc[1])  # crop features

	# np branch
	np_decoder = decoder(enc)
	np_decoder = _relu(np_decoder)
	np_decoder = Conv2D(2, 1, strides=(1,1), padding='valid', activation='softmax', name='np')(np_decoder)

	# hv branch
	hv_decoder = decoder(enc)
	hv_decoder = _relu(hv_decoder)
	hv_decoder = Conv2D(2, 1, strides=(1,1), padding='valid', name='hv')(hv_decoder)   # ?

	model = Model(inputs, outputs=[np_decoder, hv_decoder])
	return model


if __name__=="__main__":
	net = hvnet((270,270), 3)
	net.compile(loss=['categorical_crossentropy', 'mse'],
				optimizer=Adam(1e-4))


