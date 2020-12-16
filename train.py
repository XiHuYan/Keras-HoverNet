import json
import os
from os.path import join
import argparse
import keras
import keras.backend as K
from dotmap import DotMap
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler

# import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from HoverNet import *
from loss import *
from datasets import Kumar

def scheduler(epoch, lr):
    if epoch%60==0 and epoch>0:
        return lr*0.1
    else:
        return lr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_id')
    parser.add_argument(
        '--gid',
        help='gpu id')
    parser.add_argument(
        '--ds', 
        help='dataset name')
    parser.add_argument(
        '--lr',
        help='learning rate')
    parser.add_argument(
        '--eps',
        help='epochs')
    parser.add_argument(
        '--bs',
        help='batch size')

    args = parser.parse_args()
    with open('configs.json', 'r') as f:
        config = json.load(f)
    config = DotMap(config)

    # set gpu environ
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gid

    ckps_dir = join('logs/%s/%s' % (args.ds, args.exp_id))  # logs/kumar/v1.0
    callbacks = []
    callbacks.append(
        ModelCheckpoint(filepath=join(ckps_dir, "model_{epoch:02d}_{val_loss:.4f}.hdf5"),
                        monitor='val_loss',
                        save_best_only=False,
                        save_weights_only=True))
    callbacks.append(
        TensorBoard(log_dir=ckps_dir,
                    write_graph=False
        ))
    # callbacks.append(
    #     LearningRateScheduler(scheduler)
    #     )

    input_size = config.model.input_size
    input_chnl = config.model.input_chnl
    lr = float(args.lr)
    eps = int(args.eps)
    net = hvnet((input_size, input_size), input_chnl)

    net.compile(loss={'np':cce, 'hv':gmse(5)},
                optimizer=Adam(lr),
                ) # metrics={"np":[cce, soft_dice], "hv":[mse, gmse]}

    # join the parameters
    config.data_dir = join(config.data_dir, args.ds)
    config.train.batch_size = int(args.bs)

    # create the generator
    train_gen = Kumar(config, 'train')
    valid_gen = Kumar(config, 'valid')
    net.fit_generator(train_gen, 
                      steps_per_epoch=train_gen.__len__(),
                      epochs=eps,
                      validation_data=valid_gen,
                      validation_steps=valid_gen.__len__(),
                      callbacks=callbacks)










