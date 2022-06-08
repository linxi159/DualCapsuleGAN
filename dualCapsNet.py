# -*- coding: utf-8 -*-

"""
License: Apache-2.0
Author: Meng Huang
E-mail: hmlinxi@163.com
"""
import tensorflow as tf
import numpy as np

from ops import lrelu, conv2d_1, conv2d_2, batch_norm
from utils import reduce_sum, softmax
from config import FLAGS


class ConvLayer1(object):
    def __init__(self, input_=None, output_dim=512, kernel_size=9):
        # input_ is  [batch_size, 32, 32, self.df_dim*4]
        assert input_.get_shape() == [FLAGS.batch_size, 32, 32, 64*4]
        
        self.input_ = input_
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = 1
        
        self.conv1 = tf.contrib.layers.conv2d(self.input_, self.output_dim, self.kernel_size, 
                                              self.stride, padding="VALID",activation_fn=None)
          
    def forward(self):
        return self.conv1

"""
class ConvLayer1(object):
    def __init__(self, input_=None, output_dim=256, kernel_size=7):
        # input_ is  [batch_size, 128, 128, input_c_dim]
        assert input_.get_shape() == [FLAGS.batch_size, 128, 128, 3]
        
        self.input_ = input_
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        
        self.conv1 = conv2d_1(input_=self.input_, output_dim=self.output_dim, kernal_size=7)
        
    def forward(self):
        return lrelu(self.conv1)


class ConvLayer2(object):
    def __init__(self, input_=None, output_dim=256, kernel_size=9):
        # input_ is  [batch_size, 32, 32, 256]
        assert input_.get_shape() == [FLAGS.batch_size, 32, 32, 256]
        self.input_ = input_
        self.output_dim = output_dim 
        self.kernel_size = kernel_size
#        self.stride = 1
        self.conv2 = conv2d_2(input_=self.input_, output_dim=self.output_dim, kernal_size=9)
        
#        self.conv2 = tf.contrib.layers.conv2d(self.input_, self.output_dim, self.kernel_size, 
#                                              self.stride, padding="SAME",activation_fn=None)
    
    def forward(self):
        return lrelu(self.conv2)
"""  

# the PrimaryCaps layer, a convolutional layer
class PrimaryCaps(object):
    def __init__(self, input_=None, vec_len=8, num_outputs=64, output_dim=64 * 8 * 8, kernel_size=9, stride=2):
        # input_ is  [batch_size, 24, 24, 512]
        assert input_.get_shape() == [FLAGS.batch_size, 24, 24, 512]        
        self.input_ = input_
        self.output_dim = output_dim 
        self.kernel_size = kernel_size
        self.vec_len = vec_len
        self.num_outputs = num_outputs
        self.stride = stride
        

    def forward(self):
        # version 2, equivalent to version 1 but higher computational
        # efficiency.
        # NOTE: I can't find out any words from the paper whether the
        # PrimaryCap convolution does a ReLU activation or not before
        # squashing function, but experiment show that using ReLU get a
        # higher test accuracy. So, which one to use will be your choice
        self.capsules = tf.contrib.layers.conv2d(self.input_, self.num_outputs * self.vec_len,
                                            self.kernel_size, self.stride, padding="VALID",
                                            activation_fn=tf.nn.relu)
        # capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
        #                                    self.kernel_size, self.stride, padding="VALID",
        #                                    activation_fn=None)
        self.capsules = tf.reshape(self.capsules, (FLAGS.batch_size, -1, self.vec_len, 1))
        # [batch_size, 4096, 8, 1]
        self.capsules = self.squash()
        assert self.capsules.get_shape() == [FLAGS.batch_size, 4096, 8, 1]
        return(self.capsules)        
        
    def squash(self):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        '''
        self.vec_squared_norm = reduce_sum(tf.square(self.capsules), -2, keepdims=True)
        self.scalar_factor =  tf.sqrt(self.vec_squared_norm) / (1 +  tf.sqrt(self.vec_squared_norm)) / tf.sqrt(self.vec_squared_norm + FLAGS.epsilon)
        self.vec_squashed = self.scalar_factor * self.capsules  # element-wise
        return(self.vec_squashed)


# the DigitCaps layer, a fully connected layer
class DigitCaps(object):
    def __init__(self, input_=None, vec_len=16, num_outputs=1, num_routes=64 * 8 * 8, out_channels=16, with_routing=True):
        # input_ is [batch_size, 4096, 8, 1]
        assert input_.get_shape() == [FLAGS.batch_size, 4096, 8, 1]       
        self.input_ = input_
        self.num_outputs = num_outputs
        self.vec_len = vec_len
#        self.out_channels = out_channels
        self.num_routes = num_routes
        self.with_routing = with_routing
        
    def forward(self):
        
        if self.with_routing:
            # Reshape the input into [batch_size, 4096, 1, 8, 1]
            self.input = tf.reshape(self.input_, shape=(FLAGS.batch_size, -1, 1, self.input_.shape[-2].value, 1))
            assert self.input.get_shape() == [FLAGS.batch_size, 4096, 1, 8, 1] 
            
            with tf.variable_scope('routing'):
                # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                # about the reason of using 'batch_size', see issue #21
                self.b_IJ = tf.constant(np.zeros([FLAGS.batch_size, self.input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                self.capsules = self.routing()
                self.capsules = tf.squeeze(self.capsules, axis=1)

        # DigitCaps layer, return [batch_size, 1, 16, 1]
        assert self.capsules.get_shape() == [FLAGS.batch_size, 1, 16, 1] 
        return(self.capsules) 

    
    def routing(self): # self.input,self.b_IJ 
        ''' The routing algorithm.
        Args:
            input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
        Returns:
            A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
            representing the vector output `v_j` in the layer l+1
        Notes:
            u_i represents the vector output of capsule i in the layer l, and
            v_j the vector output of capsule j in the layer l+1.
        '''
        # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
        W = tf.get_variable('Weight', shape=(1, 4096, 16, 8, 1), dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=FLAGS.stddev))
        biases = tf.get_variable('bias', shape=(1, 1, 1, 16, 1))

        # Eq.2, calc u_hat
        # Since tf.matmul is a time-consuming op,
        # A better solution is using element-wise multiply, reduce_sum and reshape
        # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
        # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
        # reshape to [a, c]
        self.input = tf.tile(self.input, [1, 1, 16, 1, 1])
        assert self.input.get_shape() == [FLAGS.batch_size, 4096, 16, 8, 1]
        
        self.u_hat = reduce_sum(W * self.input, axis=3, keepdims=True)
        self.u_hat = tf.reshape(self.u_hat, shape=[-1, 4096, 1, 16, 1])
        assert self.u_hat.get_shape() == [FLAGS.batch_size, 4096, 1, 16, 1]
    
        # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
        self.u_hat_stopped = tf.stop_gradient(self.u_hat, name='stop_gradient')
        
        # line 3,for r iterations do
        for r_iter in range(FLAGS.iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                # line 4:
                # => [batch_size, 4096, 1, 1, 1]
                self.c_IJ = softmax(self.b_IJ, axis=2)
                
                # At last iteration, use `u_hat` in order to receive gradients from the following graph
                if r_iter == FLAGS.iter_routing - 1:
                    # line 5:
                    # weighting u_hat with c_IJ, element-wise in the last two dims
                    # => [batch_size, 4096, 1, 16, 1]
                    self.s_J = tf.multiply(self.c_IJ, self.u_hat)
                    # then sum in the second dim, resulting in [batch_size, 1, 1, 16, 1]
                    self.s_J = reduce_sum(self.s_J, axis=1, keepdims=True) + biases
                    assert self.s_J.get_shape() == [FLAGS.batch_size, 1, 1, 16, 1]
                    
                    # line 6:
                    # squash using Eq.1,
                    self.v_J = self.squash() # self.s_J
                    assert self.v_J.get_shape() == [FLAGS.batch_size, 1, 1, 16, 1]
                elif r_iter < FLAGS.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                    self.s_J = tf.multiply(self.c_IJ, self.u_hat_stopped)
                    self.s_J = reduce_sum(self.s_J, axis=1, keepdims=True) + biases
                    self.v_J = self.squash() # self.s_J
                    
                    # line 7:
                    # reshape & tile v_j from [batch_size ,1, 1, 16, 1] to [batch_size, 4096, 1, 16, 1]
                    # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                    # batch_size dim, resulting in [1, 4096, 1, 1, 1]
                    self.v_J_tiled = tf.tile(self.v_J, [1, 4096, 1, 1, 1])
                    self.u_produce_v = reduce_sum(self.u_hat_stopped * self.v_J_tiled, axis=3, keepdims=True)
                    assert self.u_produce_v.get_shape() == [FLAGS.batch_size, 4096, 1, 1, 1]
                    
                    # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                    self.b_IJ += self.u_produce_v

        return(self.v_J)      
        
    def squash(self):
        '''Squashing function corresponding to Eq. 1
        Args:
            vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
        Returns:
            A tensor with the same shape as vector but squashed in 'vec_len' dimension.
        '''
        self.vec_squared_norm = reduce_sum(tf.square(self.s_J), -2, keepdims=True)
        self.scalar_factor = tf.sqrt(self.vec_squared_norm) / (1 + tf.sqrt(self.vec_squared_norm)) / tf.sqrt(self.vec_squared_norm + FLAGS.epsilon)
        self.vec_squashed = self.scalar_factor * self.s_J  # element-wise
        return(self.vec_squashed)


