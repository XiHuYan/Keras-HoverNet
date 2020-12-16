import os
from os.path import join
import skimage
import skimage.morphology as morph
import skimage.io as io
import numpy as np
from augs import color_aug, shape_aug, center_crop
from utils import get_hv_dist
import keras

class Kumar(keras.utils.Sequence):
    def __init__(self, config, cv):
        self.cv = cv
        self.config = config
        self.input_size = config.model.input_size
        self.output_size = config.model.output_size
        self.iids = np.array(next(os.walk(join(config.data_dir, cv, 'images')))[2])
        self.index = np.random.permutation(len(self.iids))

    def __len__(self):
        return max(1, len(self.iids)//self.config.train.batch_size)

    def on_epoch_end(self):
        self.index = np.random.permutation(self.index)

    def train_aug(self, img, lbl, input_shape):
        img, lbl = img.astype('uint8'), lbl.astype('uint8')
        img, lbl = shape_aug(img, lbl, input_shape)
        img = color_aug(img)
        return img, lbl

    def valid_aug(self, img, lbl, input_shape):
        img, lbl = img.astype('uint8'), lbl.astype('uint8')
        img = center_crop(img, input_shape)
        lbl = center_crop(lbl, input_shape)
        return img, lbl

    def process(self, img, lbl):
        img = skimage.img_as_float(img)
        lbl = morph.remove_small_objects(lbl, 20)   # in case bias from augmentation 
        hv = get_hv_dist(lbl)
        bp = np.zeros(lbl.shape+(2,))
        bp[...,1] = (lbl>0) * 1
        bp[...,0] = 1 - bp[...,1]
        return img, hv, bp

    def crop_center(self, img, size):
        m, n = img.shape
        rs, cs = (m-size)//2, (n-size)//2
        return img[rs:(rs+size), cs:(cs+size)]

    def __getitem__(self, i):
        iids = self.iids[self.index[i*self.config.train.batch_size:(i+1)*self.config.train.batch_size]]
        X, NP, HV = [], [], []
        
        # debug = []
        for idx, iid in enumerate(iids):
            arr = np.load(join(self.config.data_dir, self.cv, 'images', iid))
            img, lbl = arr[...,:3], arr[...,3]
            aug_func = self.train_aug if self.cv=='train' else self.valid_aug
            img, lbl = aug_func(img, lbl, (self.input_size, self.input_size))

            lbl_80 = center_crop(lbl, (self.output_size, self.output_size))
            img, hv, bp = self.process(img, lbl_80)

            X.append(np.expand_dims(img, 0))
            NP.append(np.expand_dims(bp, 0))
            HV.append(np.expand_dims(hv, 0))

        X = np.vstack(X)
        NP = np.vstack(NP) 
        HV = np.vstack(HV)
        return X, [NP, HV]

if __name__ == '__main__':
    import json
    from dotmap import DotMap
    with open('configs.json', 'r') as f:
        config = json.load(f)

    config = DotMap(config)
    config.data_dir = join(config.data_dir, 'kumar')
    config.train.batch_size = 2

    train_gen = Kumar(config, 'train')
    for i in range(train_gen.__len__()):
        x, [NP, HV] = train_gen.__getitem__(i)
        print(x.shape, NP.shape, HV.shape)
        break







        


