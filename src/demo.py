# Author: Bichen Wu (bichen@berkeley.edu) 03/07/2017

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Author: Xuanyu Zhou (xuanyu_zhou@berkeley.edu), Bichen Wu (bichen@berkeley.edu) 10/27/2018

"""Base Model configurations"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

def base_model_config(dataset='KITTI'):
  assert dataset.upper()=='KITTI', \
      'Currently only support KITTI dataset'

  cfg = edict()

  # Dataset used to train/val/test model. Now support KITTI
  cfg.DATASET = dataset.upper()

  # classes
  cfg.CLASSES = [
      'unknown',
      'car',
      'van',
      'truck',
      'pedestrian',
      'person_sitting',
      'cyclist',
      'tram',
      'misc',
  ]

  # number of classes
  cfg.NUM_CLASS = len(cfg.CLASSES)

  # dict from class name to id
  cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(len(cfg.CLASSES))))

  # loss weight for each class
  cfg.CLS_LOSS_WEIGHT = np.array(
      [1/20.0, 1.0,  2.0, 3.0,
       8.0, 10.0, 8.0, 2.0, 1.0]
  )

  # rgb color for each class
  cfg.CLS_COLOR_MAP = np.array(
      [[ 0.00,  0.00,  0.00],
       [ 0.12,  0.56,  0.37],
       [ 0.66,  0.55,  0.71],
       [ 0.58,  0.72,  0.88],
       [ 0.25,  0.51,  0.76],
       [ 0.98,  0.47,  0.73],
       [ 0.40,  0.19,  0.10],
       [ 0.87,  0.19,  0.17],
       [ 0.13,  0.55,  0.63]]
  )

  # Probability to keep a node in dropout
  cfg.KEEP_PROB = 0.5

  # image width
  cfg.IMAGE_WIDTH = 224

  # image height
  cfg.IMAGE_HEIGHT = 224

  # number of vertical levels
  cfg.NUM_LEVEL = 10

  # number of pie sectors of the field of view
  cfg.NUM_SECTOR = 90

  # maximum distance of a measurement
  cfg.MAX_DIST = 100.0

  # batch size
  cfg.BATCH_SIZE = 20

  # Pixel mean values (BGR order) as a (1, 1, 3) array. Below is the BGR mean
  # of VGG16
  cfg.BGR_MEANS = np.array([[[103.939, 116.779, 123.68]]])

  # Pixel mean values (BGR order) as a (1, 1, 3) array. Below is the BGR mean
  # of VGG16
  cfg.RGB_MEANS = np.array([[[123.68, 116.779, 103.939]]])

  # reduce step size after this many steps
  cfg.DECAY_STEPS = 10000

  # multiply the learning rate by this factor
  cfg.LR_DECAY_FACTOR = 0.1

  # learning rate
  cfg.LEARNING_RATE = 0.005

  # momentum
  cfg.MOMENTUM = 0.9

  # weight decay
  cfg.WEIGHT_DECAY = 0.0005

  # wether to load pre-trained model
  cfg.LOAD_PRETRAINED_MODEL = True

  # path to load the pre-trained model
  cfg.PRETRAINED_MODEL_PATH = ''

  # print log to console in debug mode
  cfg.DEBUG_MODE = False

  # gradients with norm larger than this is going to be clipped.
  cfg.MAX_GRAD_NORM = 10.0

  # Whether to do data augmentation
  cfg.DATA_AUGMENTATION = False

  # The range to randomly shift the image widht
  cfg.DRIFT_X = 0

  # The range to randomly shift the image height
  cfg.DRIFT_Y = 0

  # small value used in batch normalization to prevent dividing by 0. The
  # default value here is the same with caffe's default value.
  cfg.BATCH_NORM_EPSILON = 1e-5

  # small value used in denominator to prevent division by 0
  cfg.DENOM_EPSILON = 1e-12

  # capacity for tf.FIFOQueue
  cfg.QUEUE_CAPACITY = 80

  cfg.NUM_ENQUEUE_THREAD = 8

  # Squeeze-Excitation parameter
  cfg.REDUCTION = 16

  cfg.TRAINING = True

  return cfg

import numpy as np


def kitti_squeezeSeg_config():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')

  mc.CLASSES            = ['unknown', 'car', 'pedestrian', 'cyclist']
  mc.NUM_CLASS          = len(mc.CLASSES)
  mc.CLS_2_ID           = dict(zip(mc.CLASSES, range(len(mc.CLASSES))))
  mc.CLS_LOSS_WEIGHT    = np.array([1/3.0, 1.0, 3.5, 3.5])
  mc.CLS_COLOR_MAP      = np.array([[ 0.00,  0.00,  0.00],
                                    [ 0.12,  0.56,  0.37],
                                    [ 0.66,  0.55,  0.71],
                                    [ 0.58,  0.72,  0.88]])

  mc.BATCH_SIZE         = 40
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64

  mc.FOCAL_GAMMA        = 2.0
  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  mc.BILATERAL_THETA_A  = np.array([.9, .9, .6, .6])
  mc.BILATERAL_THETA_R  = np.array([.015, .015, .01, .01])
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_THETA_A        = np.array([.9, .9, .6, .6])
  mc.ANG_FILTER_COEF    = 0.02

  mc.CLS_LOSS_COEF      = 15.0
  mc.WEIGHT_DECAY       = 0.0001
  mc.LEARNING_RATE      = 0.05
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance
  mc.INPUT_MEAN         = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
  mc.INPUT_STD          = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])

  return mc





from datetime import datetime
import os.path
import sys
import time
import glob    

import numpy as np
from six.moves import xrange
import tensorflow as tf
from PIL import Image

from config import *
from imdb import *
from util import *
from nets import *

FLAGS = tf.compat.v1.app.flags.FLAGS

tf.compat.v1.app.flags.DEFINE_string(
        'checkpoint', './data/SqueezeSegV2/model.ckpt-30700',
        """Path to the model parameter file.""")
tf.compat.v1.app.flags.DEFINE_string(
        'input_path', './data/samples/*',
        """Input lidar scan to be detected. Can process glob input such as """
        """./data/samples/*.npy or single input.""")
tf.compat.v1.app.flags.DEFINE_string(
        'out_dir', './data/samples_out/', """Directory to dump output.""")
tf.compat.v1.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def _normalize(x):
  return (x - x.min())/(x.max() - x.min())

def detect():
  """Detect LiDAR data."""

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():
    mc = kitti_squeezeSeg_config()
    mc.LOAD_PRETRAINED_MODEL = False
    mc.BATCH_SIZE = 1 # TODO(bichen): fix this hard-coded batch size.
    model = SqueezeSeg(mc)

    saver = tf.compat.v1.train.Saver(model.model_params)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)
      for f in glob.iglob(FLAGS.input_path):
        lidar = np.load(f).astype(np.float32, copy=False)[:, :, :5]
        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1]
        )
        lidar = (lidar - mc.INPUT_MEAN)/mc.INPUT_STD
        lidar = np.append(lidar, lidar_mask, axis=2)
        pred_cls = sess.run(
            model.pred_cls,
            feed_dict={
                model.lidar_input:[lidar],
                model.keep_prob: 1.0,
                model.lidar_mask:[lidar_mask]
            }
        )

        # save the data
        file_name = f.strip('.npy').split('/')[-1]
        np.save(
            os.path.join(FLAGS.out_dir, 'pred_'+file_name+'.npy'),
            pred_cls[0]
        )

        # save the plot
        depth_map = Image.fromarray(
            (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
        label_map = Image.fromarray(
            (255 * visualize_seg(pred_cls, mc)[0]).astype(np.uint8))

        blend_map = Image.blend(
            depth_map.convert('RGBA'),
            label_map.convert('RGBA'),
            alpha=0.4
        )

        blend_map.save(
            os.path.join(FLAGS.out_dir, 'plot_'+file_name+'.png'))


def main(argv=None):
  if not tf.io.gfile.exists(FLAGS.out_dir):
    tf.io.gfile.makedirs(FLAGS.out_dir)
  detect()
  print('Detection output written to {}'.format(FLAGS.out_dir))


if __name__ == '__main__':
    tf.compat.v1.app.run()
