# -*- coding: utf-8 -*-

"""
License: Apache-2.0
Author: Meng Huang
E-mail: hmlinxi@163.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from glob import glob
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from six.moves import xrange
from functools import partial

from ops import *
#from ops import batch_norm, conv2d, lrelu, deconv2d
from utils import *
#from utils import reduce_sum, load_checkpoint, counter, mkdir
from config import FLAGS
#from dualCapsNet import ConvLayer1, ConvLayer2, PrimaryCaps, DigitCaps
from dualCapsNet import ConvLayer1, PrimaryCaps, DigitCaps

import image_utils as im
import data

class DuCaGAN(object):
    def __init__(self, sess,image_size=256, batch_size=1, output_size=256, sample_size=1,
                 gf_dim=64, df_dim=64, input_c_dim=3, output_c_dim=3, dataset_name=' horse2zebra',
                 L1_lambda=100,log_dir=None,checkpoint_dir=None, sample_dir=None,results_dir=None):
        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        
        
        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim
        
        self.L1_lambda = L1_lambda
        
        self.dataset_name = dataset_name
        self.log_dir = log_dir + dataset_name
        self.checkpoint_dir = checkpoint_dir + dataset_name
        self.sample_dir = sample_dir + dataset_name
        self.results_dir = results_dir + dataset_name
    
        if FLAGS.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion
    
        self.real_label = 1.0
        self.fake_label = 0.0         
                
        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
 
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')
        
        self.build_model()      
                     
               
    def build_model(self):
    
        self.src_real = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.des_real = tf.placeholder(tf.float32, shape=[self.batch_size, self.image_size, self.image_size, 3])
 
#        self.real_data = tf.placeholder(tf.float32,[self.batch_size, self.image_size, self.image_size,self.input_c_dim + self.output_c_dim], name='real_src_and_des_images')
#        self.des_real = self.real_data[:, :, :, :self.input_c_dim]
#        self.src_real = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

#        self.src2des = self.generator_2(self.src_real)
        
#        self.real_srcdes = tf.concat([self.src_real, self.des_real], 3)
#        self.fake_srcdes = tf.concat([self.src_real, self.src2des], 3)
        
        
        self.fake_des = self.generator_2(self.src_real, reuse=False, name="generator_src2des")
        self.fake_src_ = self.generator_2(self.fake_des, reuse=False, name="generator_des2src")
        self.fake_src = self.generator_2(self.des_real, reuse=True, name="generator_des2src")
        self.fake_des_ = self.generator_2(self.fake_src, reuse=True, name="generator_src2des")
        
        self.fake_des_sample_ = self.sampler_2(self.src_real, name="generator_src2des")
        self.fake_src_sample_ = self.sampler_2(self.des_real, name="generator_des2src")

        self.D_des_fake_1 = self.discriminator_1(self.fake_des, reuse=False, domain="des_", class_="1")
#       self.D_des_fake_2 = self.discriminator_1(self.fake_des, reuse=False, domain="des_", class_="2")
        self.D_des_fake_2 = self.discriminator_2(self.fake_des, reuse=False, domain="des_", class_="2")
        self.D_src_fake_1 = self.discriminator_1(self.fake_src, reuse=False, domain="src_", class_="1")
#        self.D_src_fake_2 = self.discriminator_1(self.fake_src, reuse=False, domain="src_", class_="2")
        self.D_src_fake_2 = self.discriminator_2(self.fake_src, reuse=False, domain="src_", class_="2")


        # the loss of generator 
        self.g_src2des_loss_1 = self.criterionGAN(self.D_des_fake_1, tf.ones_like(self.D_des_fake_1)) +  FLAGS.lambda_v * self.margin_loss(self.D_des_fake_1, self.real_label) 
#        self.g_src2des_loss_2 = self.criterionGAN(self.D_des_fake_2, tf.ones_like(self.D_des_fake_2)) +  FLAGS.lambda_v * self.margin_loss(self.D_des_fake_2, self.real_label) 
        self.g_src2des_loss_2 = self.criterionGAN(self.D_des_fake_2, tf.ones_like(self.D_des_fake_2)) 
           
        self.g_src2des_loss = self.g_src2des_loss_1 + self.g_src2des_loss_2 + self.L1_lambda * abs_criterion(self.src_real, self.fake_src_) + self.L1_lambda * abs_criterion(self.des_real, self.fake_des_)
        
        self.g_des2src_loss_1 = self.criterionGAN(self.D_src_fake_1, tf.ones_like(self.D_src_fake_1)) + FLAGS.lambda_v * self.margin_loss(self.D_src_fake_1, self.real_label) 
#        self.g_des2src_loss_2 = self.criterionGAN(self.D_src_fake_2, tf.ones_like(self.D_src_fake_2)) + FLAGS.lambda_v * self.margin_loss(self.D_src_fake_2, self.real_label) 
        self.g_des2src_loss_2 = self.criterionGAN(self.D_src_fake_2, tf.ones_like(self.D_src_fake_2)) 
                      
        self.g_des2src_loss = self.g_des2src_loss_1 + self.g_des2src_loss_2 + self.L1_lambda * abs_criterion(self.src_real, self.fake_src_) + self.L1_lambda * abs_criterion(self.des_real, self.fake_des_)
        
        self.g_loss = self.g_src2des_loss_1 + self.g_src2des_loss_2 + self.g_des2src_loss_1 + self.g_des2src_loss_2 + self.L1_lambda * abs_criterion(self.src_real, self.fake_src_) + self.L1_lambda * abs_criterion(self.des_real, self.fake_des_)
                
        
        # the loss of discriminator  
#        self.fake_src_sample = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.input_c_dim], name='fake_src_sample')
#        self.fake_des_sample = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.output_c_dim], name='fake_des_sample')
    
        self.fake_src_sample = self.fake_src_sample_
        self.fake_des_sample = self.fake_des_sample_
    
        self.D_des_real_1 = self.discriminator_1(self.des_real, reuse=True, domain="des_", class_="1")
#        self.D_des_real_2 = self.discriminator_1(self.des_real, reuse=True, domain="des_", class_="2")
        self.D_des_real_2 = self.discriminator_2(self.des_real, reuse=True, domain="des_", class_="2")
        self.D_src_real_1 = self.discriminator_1(self.src_real, reuse=True, domain="src_", class_="1")
#        self.D_src_real_2 = self.discriminator_1(self.src_real, reuse=True, domain="src_", class_="2")
        self.D_src_real_2 = self.discriminator_2(self.src_real, reuse=True, domain="src_", class_="2")
        
        self.D_des_fake_sample_1 = self.discriminator_1(self.fake_des_sample, reuse=True, domain="des_", class_="1")
#        self.D_des_fake_sample_2 = self.discriminator_1(self.fake_des_sample, reuse=True, domain="des_", class_="2")
        self.D_des_fake_sample_2 = self.discriminator_2(self.fake_des_sample, reuse=True, domain="des_", class_="2")
        self.D_src_fake_sample_1 = self.discriminator_1(self.fake_src_sample, reuse=True, domain="src_", class_="1")
 #       self.D_src_fake_sample_2 = self.discriminator_1(self.fake_src_sample, reuse=True, domain="src_", class_="2")
        self.D_src_fake_sample_2 = self.discriminator_2(self.fake_src_sample, reuse=True, domain="src_", class_="2")
        
        self.d_des_loss_real_1 = self.criterionGAN(self.D_des_real_1, tf.ones_like(self.D_des_real_1)) + FLAGS.lambda_v * self.margin_loss(self.D_des_real_1, self.real_label) # d
#        self.d_des_loss_real_2 = self.criterionGAN(self.D_des_real_2, tf.ones_like(self.D_des_real_2)) + FLAGS.lambda_v * self.margin_loss(self.D_des_real_2, self.real_label) # d
        self.d_des_loss_real_2 = self.criterionGAN(self.D_des_real_2, tf.ones_like(self.D_des_real_2)) 
        
        self.d_des_loss_fake_1 = self.criterionGAN(self.D_des_fake_sample_1, tf.zeros_like(self.D_des_fake_sample_1)) + FLAGS.lambda_v * self.margin_loss(self.D_des_fake_sample_1, self.fake_label) # d
#        self.d_des_loss_fake_2 = self.criterionGAN(self.D_des_fake_sample_2, tf.zeros_like(self.D_des_fake_sample_2)) + FLAGS.lambda_v * self.margin_loss(self.D_des_fake_sample_2, self.fake_label) # d
        self.d_des_loss_fake_2 = self.criterionGAN(self.D_des_fake_sample_2, tf.zeros_like(self.D_des_fake_sample_2)) 

        self.d_des_loss = (self.d_des_loss_real_1 + self.d_des_loss_real_2 + self.d_des_loss_fake_1 + self.d_des_loss_fake_2) / 2 

        self.d_src_loss_real_1 = self.criterionGAN(self.D_src_real_1, tf.ones_like(self.D_src_real_1)) + FLAGS.lambda_v * self.margin_loss(self.D_src_real_1, self.real_label) # d
#        self.d_src_loss_real_2 = self.criterionGAN(self.D_src_real_2, tf.ones_like(self.D_src_real_2)) + FLAGS.lambda_v * self.margin_loss(self.D_src_real_2, self.real_label) # d
        self.d_src_loss_real_2 = self.criterionGAN(self.D_src_real_2, tf.ones_like(self.D_src_real_2)) 
                 
        self.d_src_loss_fake_1 = self.criterionGAN(self.D_src_fake_sample_1, tf.zeros_like(self.D_src_fake_sample_1)) + FLAGS.lambda_v * self.margin_loss(self.D_src_fake_sample_1, self.fake_label) # d
#        self.d_src_loss_fake_2 = self.criterionGAN(self.D_src_fake_sample_2, tf.zeros_like(self.D_src_fake_sample_2)) + FLAGS.lambda_v * self.margin_loss(self.D_src_fake_sample_2, self.fake_label) # d
        self.d_src_loss_fake_2 = self.criterionGAN(self.D_src_fake_sample_2, tf.zeros_like(self.D_src_fake_sample_2))
           
        self.d_src_loss = (self.d_src_loss_real_1 + self.d_src_loss_real_2 + self.d_src_loss_fake_1 + self.d_src_loss_fake_2) / 2 
        
        self.d_loss = self.d_des_loss + self.d_src_loss
        
    
        # summary writer
#        self.d_des_real_1_sum = tf.summary.histogram("d_des_real_1", self.D_des_real_1) # d
#        self.d_des_real_2_sum = tf.summary.histogram("d_des_real_2", self.D_des_real_2) # d
#        self.d_src_real_1_sum = tf.summary.histogram("d_src_real_1", self.D_src_real_1) # d
#        self.d_src_real_2_sum = tf.summary.histogram("d_src_real_2", self.D_src_real_2) # d
        
#        self.d_des_fake_sample_1_sum = tf.summary.histogram("d_des_fake_sample_1", self.D_des_fake_sample_1) # d
#        self.d_des_fake_sample_2_sum = tf.summary.histogram("d_des_fake_sample_2", self.D_des_fake_sample_2) # d
#        self.d_src_fake_sample_1_sum = tf.summary.histogram("d_src_fake_sample_1", self.D_src_fake_sample_1) # d
#        self.d_src_fake_sample_2_sum = tf.summary.histogram("d_src_fake_sample_2", self.D_src_fake_sample_2) # d

        self.d_des_loss_sum = tf.summary.scalar("d_des_loss", self.d_des_loss)
        self.d_src_loss_sum = tf.summary.scalar("d_src_loss", self.d_src_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

  
        self.d_des_loss_real_sum = tf.summary.scalar("d_des_loss_real", self.d_des_loss_real_1 + self.d_des_loss_real_2)
        self.d_des_loss_fake_sum = tf.summary.scalar("d_des_loss_fake", self.d_des_loss_fake_1 + self.d_des_loss_fake_2)
        self.d_src_loss_real_sum = tf.summary.scalar("d_src_loss_real", self.d_src_loss_real_1 + self.d_src_loss_real_2)
        self.d_src_loss_fake_sum = tf.summary.scalar("d_src_loss_fake", self.d_src_loss_fake_1 + self.d_src_loss_fake_2) 
        
        self.d_des_sum = tf.summary.merge([self.d_des_loss_sum, self.d_des_loss_real_sum, self.d_des_loss_fake_sum, self.d_loss_sum])
        
        self.d_src_sum = tf.summary.merge([self.d_src_loss_sum, self.d_src_loss_real_sum, self.d_src_loss_fake_sum,self.d_loss_sum])

#        self.g_src2des_sum = tf.summary.image("g_src2des", self.fake_des) # g
#        self.g_fakesrc2des_sum = tf.summary.image("g_fakesrc2des", self.fake_des_) # g
#        self.g_des2src_sum = tf.summary.image("g_des2src", self.fake_src) # g
#        self.g_fakedes2src_sum = tf.summary.image("g_fakedes2src", self.fake_src_) # g

        self.g_src2des_loss_sum = tf.summary.scalar("g_src2des_loss", self.g_src2des_loss)
        self.g_des2src_loss_sum = tf.summary.scalar("g_des2src_loss", self.g_des2src_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        
        self.g_sum = tf.summary.merge([self.g_src2des_loss_sum, self.g_des2src_loss_sum, self.g_loss_sum])
                              
        t_vars = tf.trainable_variables()

        self.d_des_vars = [var for var in t_vars if 'des_discriminator_1' in var.name or 'des_discriminator_2' in var.name]
        self.d_src_vars = [var for var in t_vars if 'src_discriminator_1' in var.name or 'src_discriminator_2' in var.name]
        self.g_vars = [var for var in t_vars if 'generator_src2des' in var.name or 'generator_des2src' in var.name]
      
        for var in t_vars: print(var.name)
        
        self.saver = tf.train.Saver(max_to_keep=5)
              
            
    def train(self, config):
        """Train DualCaps_GAN"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_src_optim = tf.train.AdamOptimizer(self.lr, beta1=config.beta1).minimize(self.d_src_loss, var_list=self.d_src_vars)
        self.d_des_optim = tf.train.AdamOptimizer(self.lr, beta1=config.beta1).minimize(self.d_des_loss, var_list=self.d_des_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)

        try:
            self.init_op = tf.global_variables_initializer()
            self.sess.run(self.init_op)
        except:
            self.init_op_ = tf.initialize_all_variables()
            self.sess.run(self.init_op_)

#        self.d_sum = tf.summary.merge([self.d_des_1_sum,self.d_des_2_sum, self.d_des_loss_sum, self.d_loss_sum])
#        self.g_sum = tf.summary.merge([self.d_src2des_1_sum, self.d_src2des_2_sum, self.g_src2des_sum, self.d_src2des_loss_sum, self.g_loss_sum])
#        self.d_sum = tf.summary.merge([self.d_des_sum,self.d_src2des_sum, self.d_des_loss_sum, self.d_src2des_loss_sum, self.d_loss_sum])
#        self.g_sum = tf.summary.merge([self.g_src2des_sum, self.g_loss_sum])
        
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        
        # init before preparing data 
        self.src_img_paths = glob(config.data_dir + config.dataset + '/trainA/*.jpg')
        self.des_img_paths = glob(config.data_dir + config.dataset + '/trainB/*.jpg')
        self.src_data_pool = data.ImageData(self.sess, self.src_img_paths, config.batch_size, load_size=config.load_size, crop_size=config.crop_size)
        self.des_data_pool = data.ImageData(self.sess, self.des_img_paths, config.batch_size, load_size=config.load_size, crop_size=config.crop_size)
        self.src_test_img_paths = glob(config.data_dir + config.dataset + '/testA/*.jpg')
        self.src_test_data_pool = data.ImageData(self.sess, self.src_test_img_paths,config.batch_size, load_size=config.load_size, crop_size=config.crop_size)
        self.des_test_img_paths = glob(config.data_dir + config.dataset + '/testB/*.jpg')
        self.des_test_data_pool = data.ImageData(self.sess, self.des_test_img_paths,config.batch_size, load_size=config.load_size, crop_size=config.crop_size)

#        src2des_pool = utils.ItemPool() 
#        des2src_pool = utils.ItemPool()            

        # loading model
        try:
            self.ckpt_path__ = self.load_checkpoint(self.checkpoint_dir)
        except:
            print(" [*] Loading checkpoint failed...")        

#        if self.load(self, self.checkpoint_dir):
#            print(" [*] Load  CheckPoint success")
#        else:
#            print(" [!] Load  CheckPoint failed...")

 
        # counter
        counter = 1
        start_time = time.time()   
        
        for epoch in xrange(config.epoch):
            batch_idxs = min(len(self.src_data_pool), len(self.des_data_pool), config.train_size) // self.batch_size
            lr = config.learning_rate if epoch < config.epoch else config.learning_rate*(config.epoch - epoch) / (config.epoch - config.epoch_step)
            
            for idx in xrange(0, batch_idxs):
                
              # preparing data
                self.src_real_ipt = self.src_data_pool.batch()
                self.des_real_ipt = self.des_data_pool.batch() 
                                                                 
                # Update G network and record fake outputs
                _, g_summary_opt = self.sess.run(
                        [self.g_optim, self.g_sum],
                        feed_dict={self.src_real: self.src_real_ipt, self.des_real: self.des_real_ipt, self.lr: lr})
                self.writer.add_summary(g_summary_opt, counter)
            
                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, g_summary_opt = self.sess.run(
                        [self.g_optim, self.g_sum],
                        feed_dict={self.src_real: self.src_real_ipt, self.des_real: self.des_real_ipt, self.lr: lr})
                self.writer.add_summary(g_summary_opt, counter) 
                
#                src2des_sample_ipt = np.array(src2des_pool(list(src2des_opt)))
#                des2src_sample_ipt = np.array(des2src_pool(list(des2src_opt)))
#                [fake_src, fake_des] = self.pool([fake_src, fake_des])
                
                # Update D_des network
                _, d_des_summary_opt = self.sess.run(
                        [self.d_des_optim, self.d_des_sum], 
                        feed_dict={self.src_real: self.src_real_ipt, self.des_real: self.des_real_ipt, self.lr: lr})
                self.writer.add_summary(d_des_summary_opt, counter) 
                
                # Update D_src network
                _, d_src_summary_opt = self.sess.run(
                        [self.d_src_optim, self.d_src_sum], 
                        feed_dict={self.src_real: self.src_real_ipt, self.des_real: self.des_real_ipt, self.lr: lr})
                self.writer.add_summary(d_src_summary_opt, counter) 

                errD = self.d_loss.eval({self.src_real: self.src_real_ipt, self.des_real: self.des_real_ipt})
                errG = self.g_loss.eval({self.src_real: self.src_real_ipt, self.des_real: self.des_real_ipt})

                counter += 1
                # display
                if counter % 1 == 0:
                    print("Epoch: [%3d] [%5d/%5d], time: %5.5f, d_loss: %.8f, g_loss: %.8f" % ((epoch, idx, batch_idxs, time.time() - start_time, errD, errG)))
                
                # save
                if (counter + 1) % 500 == 0:                    
                    self.save_path = self.saver.save(self.sess, "%s/Epoch_(%d)_(%dof%d).ckpt" % (self.checkpoint_dir, epoch, idx, batch_idxs))
                    print("Model saved in file: % s" % self.save_path)

                # sample
                if (counter + 1) % 100 == 0:
#                    sample_images = self.load_random_samples()
                    src_test_real_ipt = self.src_test_data_pool.batch()
                    des_test_real_ipt = self.des_test_data_pool.batch()
                    
                    [src2des_opt, src2des2src_opt, des2src_opt, des2src2des_opt] = self.sess.run(
                            [self.fake_des, self.fake_src_, self.fake_src, self.fake_des_],
                            feed_dict={self.src_real: src_test_real_ipt, self.des_real: des_test_real_ipt})
                    
                    sample_opt = np.concatenate((src_test_real_ipt, src2des_opt, src2des2src_opt, des_test_real_ipt, des2src_opt, des2src2des_opt), axis=0)
                    
                    im.imwrite(im.immerge(sample_opt, 2, 3), '%s/Epoch_(%d)_(%dof%d).jpg' % (self.sample_dir, epoch, idx, batch_idxs))

                # sample
                if (counter + 1) % 5000 == 0:
                    # init before testing        
                    self.src_test_img_list = glob(config.data_dir + config.dataset + '/testA/*.jpg')
                    self.des_test_img_list = glob(config.data_dir + config.dataset + '/testB/*.jpg')
       
                    counter_ = '%d' %counter
                    self.src_save_dir = self.results_dir + '/' + counter_ + '/test_src'
                    self.des_save_dir = self.results_dir + '/' + counter_ + '/test_des'
        
                    mkdir([self.src_save_dir])
                    mkdir([self.des_save_dir])
        
                    self.start_time_ = time.time()
        
                    for i in range(len(self.src_test_img_list)):
                        # load testing input
                        print("Loading src_testing images ...")
                        src_test_real_ipt_ = im.imresize(im.imread(self.src_test_img_list[i]), [config.crop_size, config.crop_size])
                        src_test_real_ipt_.shape = 1, config.crop_size, config.crop_size, 3
                        src2des_opt, src2des2src_opt = self.sess.run([self.fake_des, self.fake_src_], feed_dict={self.src_real: src_test_real_ipt_})
                        src_img_opt = np.concatenate((src_test_real_ipt_,src2des_opt, src2des2src_opt), axis=0)

                        img_name = os.path.basename(self.src_test_img_list[i])
                        im.imwrite(im.immerge(src_img_opt, 1, 3), self.src_save_dir + '/' + img_name)
                        print('Save %s' % (self.src_save_dir + '/' + img_name))

                    for i in range(len(self.des_test_img_list)):
                        # load testing input
                        print("Loading des_testing images ...")
                        des_test_real_ipt_ = im.imresize(im.imread(self.des_test_img_list[i]), [config.crop_size, config.crop_size])
                        des_test_real_ipt_.shape = 1, config.crop_size, config.crop_size, 3
                        des2src_opt, des2src2des_opt = self.sess.run([self.fake_src, self.fake_des_], feed_dict={self.des_real: des_test_real_ipt_})
                        des_img_opt = np.concatenate((des_test_real_ipt_, des2src_opt, des2src2des_opt), axis=0)
            
                        img_name = os.path.basename(self.des_test_img_list[i])
                        im.imwrite(im.immerge(des_img_opt, 1, 3), self.des_save_dir + '/' + img_name)
                        print('Save %s' % (self.des_save_dir + '/' + img_name))

    def load(self, checkpoint_dir):          
        print(' [*] Loading checkpoint...')
        
        if os.path.isdir(checkpoint_dir):
            checkpoint_dir_or_file = tf.train.latest_checkpoint(checkpoint_dir)
        
        self.saver.restore(self.sess, checkpoint_dir_or_file)

        print(' [*] Loading succeeds! Copy variables from % s' % checkpoint_dir_or_file)
        
        
    def load_checkpoint(self, ckpt_dir_or_file, var_list=None):
        """Load checkpoint.

           Note:
               This function add some useless ops to the graph. It is better
               to use tf.train.init_from_checkpoint(...).
        """
        print(' [*] Loading checkpoint...')
        if os.path.isdir(ckpt_dir_or_file):
           ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

#        restorer = tf.train.Saver(var_list)
        self.saver.restore(self.sess, ckpt_dir_or_file)
        print(' [*] Loading succeeds! Copy variables from % s' % ckpt_dir_or_file)
              
              
    def discriminator_1(self, image, y=None, reuse=False, domain=None, class_=None, name="discriminator_"):
        
        with tf.variable_scope(domain + 'discriminator_' + class_) as scope:
            # image is FLAGS.batch_size x 256 x 256 x input_c_dim
            assert image.get_shape() == [FLAGS.batch_size, 256, 256, 3]
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False
                
            # build net of discriminator
            # h0 is (128 x 128 x self.df_dim)
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h1 is (64 x 64 x self.df_dim*2)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            
            assert h2.get_shape() == [FLAGS.batch_size, 32, 32, self.df_dim*4]
                                   
            conv_layer1 = ConvLayer1(input_=h2)
            conv_layer1_out_ = conv_layer1.forward()
            
            conv_layer1_out = lrelu(self.d_bn3(conv_layer1_out_))
               
#            conv_layer2 = ConvLayer2(input_= conv_layer1_out)
#            conv_layer2_out = conv_layer2.forward()
            
            primary_caps = PrimaryCaps(input_=conv_layer1_out)
            primary_caps_out = primary_caps.forward()
            
            digit_caps = DigitCaps(input_=primary_caps_out)
            digit_caps_out = digit_caps.forward()
            
            assert digit_caps_out.get_shape() == [FLAGS.batch_size, 1, 16, 1]
      
            self.prediction_capsule_length = tf.sqrt(reduce_sum(tf.square(digit_caps_out), axis=2, keepdims=True) + FLAGS.epsilon)
                   
        #predicts probability vector of one output vector[FLAGS.batch_size, 1, 1, 1]
        return self.prediction_capsule_length
    
    
    def discriminator_2(self, image, y=None, reuse=False, dim=64, train=True, domain=None, class_=None, name="discriminator_"):
        
        with tf.variable_scope(domain + 'discriminator_' + class_) as scope:
            # image is FLAGS.batch_size x 256 x 256 x input_c_dim 
            assert image.get_shape() == [FLAGS.batch_size, 256, 256, 3]
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False

            conv = partial(slim.conv2d, activation_fn=None)
            deconv = partial(slim.conv2d_transpose, activation_fn=None)
            relu = tf.nn.relu
            lrelu = partial(tf.nn.leaky_relu, alpha=0.2)
            batch_norm = partial(slim.batch_norm, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)

            bn = partial(batch_norm, is_training=train)
            conv_bn_lrelu = partial(conv, normalizer_fn=bn, activation_fn=lrelu, biases_initializer=None)

            net = lrelu(conv(image, dim, 4, 2))
            net = conv_bn_lrelu(net, dim * 2, 4, 2)
            net = conv_bn_lrelu(net, dim * 4, 4, 2)
            net = conv_bn_lrelu(net, dim * 8, 4, 1)
            net = conv(net, 1, 4, 1)

        return net

      
    def margin_loss(self, vector_len, label):
        
        #predicts probability vector of one output vector[FLAGS.batch_size, 1, 1, 1]
        self.vector_len = vector_len
        assert self.vector_len.get_shape() == [FLAGS.batch_size, 1, 1, 1]
        
        label_ = tf.reshape(label, shape=(FLAGS.batch_size, -1))
        assert label_.get_shape() == [FLAGS.batch_size, 1]
      
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., FLAGS.m_plus - self.vector_len))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.vector_len - FLAGS.m_minus))
        assert max_l.get_shape() == [FLAGS.batch_size, 1, 1, 1]
        assert max_l.get_shape() == [FLAGS.batch_size, 1, 1, 1]

        # reshape: [batch_size, 1, 1, 1] => [batch_size, 1]
        max_l = tf.reshape(max_l, shape=(FLAGS.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(FLAGS.batch_size, -1))  

        # calc loss: [batch_size, 1]
        # [batch_size, 1], element-wise multiply
        loss = label_ * max_l + FLAGS.lambda_val * (1.0 - label_) * max_r
        assert loss.get_shape() == [FLAGS.batch_size, 1]
        
        self.margin_loss_ = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        
        return self.margin_loss_
                   
                   
    def generator_1(self, image, y=None, reuse=False, name="generator"):
          
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False
                            
            # image is FLAGS.batch_size x 256 x 256 x input_c_dim
            assert image.get_shape() == [FLAGS.batch_size, 256, 256, 3]
            
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            
            # image is (FLAGS.batch_size x 256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            assert self.d8.get_shape() == [FLAGS.batch_size, 256, 256, 3]

            return tf.nn.tanh(self.d8)
        
        
    def generator_2(self, image, y=None, dim=64, reuse=False, train=True, name="generator"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()
            else:
                assert scope.reuse is False
            # image is FLAGS.batch_size x 256 x 256 x input_c_dim
            assert image.get_shape() == [FLAGS.batch_size, 256, 256, 3]
            
            conv = partial(slim.conv2d, activation_fn=None)
            deconv = partial(slim.conv2d_transpose, activation_fn=None)
            relu = tf.nn.relu
            lrelu = partial(tf.nn.leaky_relu, alpha=0.2)
            batch_norm = partial(slim.batch_norm, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)
       
            bn = partial(batch_norm, is_training=train)
            conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
            deconv_bn_relu = partial(deconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

            def _residule_block(x, dim):
                y = conv_bn_relu(x, dim, 3, 1)
                y = bn(conv(y, dim, 3, 1))
                return y + x
        
            net = conv_bn_relu(image, dim, 7, 1)
            net = conv_bn_relu(net, dim * 2, 3, 2)
            net = conv_bn_relu(net, dim * 4, 3, 2)

            for i in range(9):
                net = _residule_block(net, dim * 4)

            net = deconv_bn_relu(net, dim * 2, 3, 2)
            net = deconv_bn_relu(net, dim, 3, 2)
            net = conv(net, 3, 7, 1)
            net = tf.nn.tanh(net)
            
            return net
           
        
    def sampler_1(self, image, y=None, name='generator'):
        
        with tf.variable_scope(name) as scope:
            scope.reuse_variables()
            # image is FLAGS.batch_size x 256 x 256 x input_c_dim
            assert image.get_shape() == [FLAGS.batch_size, 256, 256, 3]
            
            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            
            # image is (FLAGS.batch_size x 256 x 256 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.gf_dim*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.gf_dim*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.gf_dim*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.gf_dim*8)

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d2 is (4 x 4 x self.gf_dim*8*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d3 is (8 x 8 x self.gf_dim*8*2)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)
            # d8 is (256 x 256 x output_c_dim)

            assert self.d8.get_shape() == [FLAGS.batch_size, 256, 256, 3]

            return tf.nn.tanh(self.d8)
            
        
    def sampler_2(self, image, y=None, dim=64, train=True, name='generator'):
        
        with tf.variable_scope(name) as scope:
            scope.reuse_variables()
            # image is FLAGS.batch_size x 256 x 256 x input_c_dim
            assert image.get_shape() == [FLAGS.batch_size, 256, 256, 3]
            
            conv = partial(slim.conv2d, activation_fn=None)
            deconv = partial(slim.conv2d_transpose, activation_fn=None)
            relu = tf.nn.relu
            lrelu = partial(tf.nn.leaky_relu, alpha=0.2)
            batch_norm = partial(slim.batch_norm, scale=True, decay=0.9, epsilon=1e-5, updates_collections=None)
       
            bn = partial(batch_norm, is_training=train)
            conv_bn_relu = partial(conv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)
            deconv_bn_relu = partial(deconv, normalizer_fn=bn, activation_fn=relu, biases_initializer=None)

            def _residule_block(x, dim):
                y = conv_bn_relu(x, dim, 3, 1)
                y = bn(conv(y, dim, 3, 1))
                return y + x
        
            net = conv_bn_relu(image, dim, 7, 1)
            net = conv_bn_relu(net, dim * 2, 3, 2)
            net = conv_bn_relu(net, dim * 4, 3, 2)

            for i in range(9):
                net = _residule_block(net, dim * 4)

            net = deconv_bn_relu(net, dim * 2, 3, 2)
            net = deconv_bn_relu(net, dim, 3, 2)
            net = conv(net, 3, 7, 1)
            net = tf.nn.tanh(net)
            
            return net
                

    def test(self, config):
        """Test DuCaGAN"""
        self.init_op_ = tf.global_variables_initializer()
        self.sess.run(self.init_op_)

        # init before testing        
        self.src_test_img_list = glob(config.data_dir + config.dataset + '/testA/*.jpg')
        self.des_test_img_list = glob(config.data_dir + config.dataset + '/testB/*.jpg')
       
        self.src_save_dir = self.results_dir + '/test_src'
        self.des_save_dir = self.results_dir + '/test_des'
        
        mkdir([self.src_save_dir])
        mkdir([self.des_save_dir])
        
        self.start_time_ = time.time()
        # retore
        try:
            self.ckpt_path_ = self.load_checkpoint(self.checkpoint_dir)
        except:
            raise Exception("No checkpoint[!] Train a model first, then run test mode")  
            
#        if self.load(self.checkpoint_dir):
#            print(" [*] Load  CheckPoint success")
#        else:
#            print(" [!] Load  CheckPoint failed,No checkpoint[!] Train a model first, then run test mode...")
   
        for i in range(len(self.src_test_img_list)):
            # load testing input
            print("Loading src_testing images ...")
            src_test_real_ipt_ = im.imresize(im.imread(self.src_test_img_list[i]), [config.crop_size, config.crop_size])
            src_test_real_ipt_.shape = 1, config.crop_size, config.crop_size, 3
            src2des_opt, src2des2src_opt = self.sess.run([self.fake_des, self.fake_src_], feed_dict={self.src_real: src_test_real_ipt_})
            src_img_opt = np.concatenate((src_test_real_ipt_,src2des_opt, src2des2src_opt), axis=0)

            img_name = os.path.basename(self.src_test_img_list[i])
            im.imwrite(im.immerge(src_img_opt, 1, 3), self.src_save_dir + '/' + img_name)
            print('Save %s' % (self.src_save_dir + '/' + img_name))

        for i in range(len(self.des_test_img_list)):
            # load testing input
            print("Loading des_testing images ...")
            des_test_real_ipt_ = im.imresize(im.imread(self.des_test_img_list[i]), [config.crop_size, config.crop_size])
            des_test_real_ipt_.shape = 1, config.crop_size, config.crop_size, 3
            des2src_opt, des2src2des_opt = self.sess.run([self.fake_src, self.fake_des_], feed_dict={self.des_real: des_test_real_ipt_})
            des_img_opt = np.concatenate((des_test_real_ipt_, des2src_opt, des2src2des_opt), axis=0)
            
            img_name = os.path.basename(self.des_test_img_list[i])
            im.imwrite(im.immerge(des_img_opt, 1, 3), self.des_save_dir + '/' + img_name)
            print('Save %s' % (self.des_save_dir + '/' + img_name))

