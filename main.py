# -*- coding: utf-8 -*-

"""
License: Apache-2.0
Author: Meng Huang
E-mail: hmlinxi@163.com
"""

import os
import tensorflow as tf

from model import DuCaGAN
from utils import pp , show_all_variables
from config import flags, FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.log_dir+FLAGS.dataset):
        os.makedirs(FLAGS.log_dir+FLAGS.dataset)
    if not os.path.exists(FLAGS.sample_dir+FLAGS.dataset):
        os.makedirs(FLAGS.sample_dir+FLAGS.dataset)   
    if not os.path.exists(FLAGS.checkpoint_dir+FLAGS.dataset):
        os.makedirs(FLAGS.checkpoint_dir+FLAGS.dataset)
    if not os.path.exists(FLAGS.results_dir+FLAGS.dataset):
        os.makedirs(FLAGS.results_dir+FLAGS.dataset)
      
    # GPU--config
    run_config = tf.ConfigProto(allow_soft_placement=True)
    run_config.gpu_options.allow_growth=True
    
    with tf.Session(config=run_config) as sess:   
        tf.logging.info(' Loading Graph[model]...')
        model =  DuCaGAN(sess, image_size=FLAGS.crop_size, batch_size=FLAGS.batch_size, dataset_name=FLAGS.dataset,
                         log_dir=FLAGS.log_dir, checkpoint_dir=FLAGS.checkpoint_dir, 
                         sample_dir=FLAGS.sample_dir,results_dir = FLAGS.results_dir)
        tf.logging.info(' Graph[model] loaded')
        
        show_all_variables()

#        sv = tf.train.Supervisor(graph=model.graph, logdir=cfg.logdir, save_model_secs=0)
        
        if FLAGS.is_training:
            tf.logging.info(' Start training...')
            model.train(FLAGS)  
            tf.logging.info(' Training done')
        else:
            model.test(FLAGS)         
      
if __name__ == '__main__':
    tf.app.run()
