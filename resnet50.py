from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Activation, BatchNormalization
from keras.optimizers import Adam, SGD
import keras.backend as K
import numpy as np

def _relu(inputs):
    return Activation('relu')(inputs)

def _bn_relu(inputs):
    bn = BatchNormalization()(inputs)
    relu = Activation('relu')(bn)
    return relu

def _conv_relu(out_dim, ks, strides, padding):
    def f(inputs):
        conv = Conv2D(out_dim, ks, strides=strides, padding=padding)(inputs)
        conv = _relu(conv)
        return conv
    return f

def _conv_bn_relu(out_dim, ks, strides, padding):
    def f(inputs):
        conv = Conv2D(out_dim, ks, strides=strides, padding=padding)(inputs)
        conv = _bn_relu(conv)
        return conv
    return f

# resnet50 unit
def _res_unit(n_in, n_mid, n_out, s, inputs, _conv_blk=_conv_relu, _skip_act=_relu):
    conv = _conv_blk(n_mid, 1, strides=(1,1), padding='same')(inputs)
    conv = _conv_blk(n_mid, 3, strides=(s,s), padding='same')(conv)
    conv = Conv2D(n_out, 1, strides=(1,1), padding='same')(conv)

    if n_in!=n_out:
        shortcut = Conv2D(n_out, 1, strides=(s,s), padding='same')(inputs)
    else:
        shortcut = inputs
    conv = Add()([conv, shortcut])  # all the operation should be implemented as layers
    conv = _skip_act(conv)
    return conv

def _res_stage(n_in, n_mid, n_out, first_strides, inputs, n_units, _conv_blk=_conv_relu, _skip_act=_relu):
    unit = _res_unit(n_in, n_mid, n_out, first_strides, inputs, _conv_blk, _skip_act)
    for i in range(n_units-1):
#         print('stage %s' % str(i))
        unit = _res_unit(n_out, n_mid, n_out, 1, unit, _conv_blk, _skip_act)
    return unit

def resnet50(in_shape=(224,224), n_chnl=3, n_class=1000):
    base_chnl = 64
    stage_chls = np.arange(1, 5, 1) * base_chnl
    stage_units = [3,4,6,3]
    inputs = Input(in_shape+(n_chnl,))

    # conv1
    conv1 = _conv_relu(base_chnl, 7, strides=(2,2), padding='same')(inputs)

    # pool1 
    pool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(conv1)

    # stage 2
    stage1 = _res_stage(stage_chls[0],   stage_chls[0], stage_chls[0]*4, 1, pool1, stage_units[0])
    stage2 = _res_stage(stage_chls[0]*4, stage_chls[1], stage_chls[1]*4, 2, stage1, stage_units[1])
    stage3 = _res_stage(stage_chls[1]*4, stage_chls[2], stage_chls[2]*4, 2, stage2, stage_units[2])
    stage4 = _res_stage(stage_chls[2]*4, stage_chls[3], stage_chls[3]*4, 2, stage3, stage_units[3])

    globalpooling = GlobalAveragePooling2D()(stage4)
    output = Dense(n_class, activation='softmax')(globalpooling)
    
    model = Model(inputs=inputs, outputs=output)
    return model

