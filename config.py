# -*- coding: utf-8 -*-

"""
License: Apache-2.0
Author: Meng Huang
E-mail: hmlinxi@163.com
"""

import tensorflow as tf
import numpy as np

flags = tf.app.flags

############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float("m_plus", 0.9, "the parameter of m plus[0.9]")
flags.DEFINE_float("m_minus", 0.1, "the parameter of m minus[0.1]")
flags.DEFINE_float("lambda_val", 0.5, "down weight of the loss for absent digit classes[0.5]")
flags.DEFINE_float("lambda_v", 0.5, "down weight of the margin loss[0.5]")

# for training
flags.DEFINE_integer("epoch",86, "Epoch to train [200]")
flags.DEFINE_integer("epoch_step", 80, "Epoch to decay lr [100]")

flags.DEFINE_integer("batch_size", 1, "The size of batch images [1]")
flags.DEFINE_integer("iter_routing", 3, "number of iterations in routing algorithm[3]")

flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")

flags.DEFINE_float("epsilon",1e-9, "error of control[1e-9]") 
flags.DEFINE_float("stddev", 0.01, "stddev for W initializer[0.01]")
flags.DEFINE_float("regularization_scale", 0.392, "regularization coefficient for reconstruction loss, default to 0.0005*784=0.392")

flags.DEFINE_float("L1_lambda",10.0, "weight on L1 term in objective[10.0]") 


############################
#   environment setting    #
############################

flags.DEFINE_integer("load_size", 286, "scale images to this size. [286]")
flags.DEFINE_integer("crop_size", 256, "then crop to this size. [256]")
flags.DEFINE_integer("train_size", 1e8, "images used to train. [1e8]")


flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("data_dir", "./datasets/", "Root directory of dataset [datasets]")
flags.DEFINE_string("dataset", "horse2zebra", "The name of dataset [horse2zebra, apple2orange]")


flags.DEFINE_boolean("is_training", True, "train or predict phase[True]")
flags.DEFINE_boolean("use_lsgan", True, "gan loss defined in lsgan[True]")

flags.DEFINE_integer("num_threads", 8, "number of threads of enqueueing examples[8]")

flags.DEFINE_string("log_dir", "./outputs/summaries/", "logs directory[summaries]")
flags.DEFINE_string("sample_dir", "./outputs/sample_images_while_training/", "Directory name to save the image samples while training[sample_images_while_training]")
flags.DEFINE_string("checkpoint_dir", "./outputs/checkpoints/", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_string("results_dir", "./outputs/test_predictions/", "path for saving results[test_predictions]")


############################
#   distributed setting    #
############################

flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

FLAGS = flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)

