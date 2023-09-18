import os
import time
import math
import tensorflow as tf
import numpy as np

class NeuralCalculation(object):
    def __init__(self):
        pass
    

    def sample_z(self, m, n):
        # Used in GAN noise generalization.
        return np.random.uniform(-1., 1., size=[m, n])   

    def spectral_norm(self, w, iteration=1):
  
        def l2_norm(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])
        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm

    def linear(self, input_, output_size, name="linear", stddev=None, spectral_normed=False, reuse=False):
        shape = input_.get_shape().as_list()

        if stddev is None:
            stddev = np.sqrt(1. / (shape[1]))

        with tf.variable_scope(name, reuse=reuse) as scope:
            weight = tf.get_variable("w", [shape[1], output_size], tf.float32, tf.truncated_normal_initializer(stddev=stddev))
            bias = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0))

            if spectral_normed:
                mul = tf.matmul(input_, self.spectral_norm(weight))
            else:
                mul = tf.matmul(input_, weight)

        return mul + bias
  
    def conv2d(self, input_, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, stddev=None, name="conv2d", spectral_normed=False, reuse=False, padding="SAME"):

        fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
        fan_out = k_h * k_w * output_dim
        if stddev is None:
            stddev = np.sqrt(2. / (fan_in))
        
        with tf.variable_scope(name, reuse=reuse) as scope:
            w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            if spectral_normed:
                conv = tf.nn.conv2d(input_, self.spectral_norm(w), strides=[1, d_h, d_w, 1], padding=padding)
            else:
                conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

            biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
            
        return conv

    def deconv2d(self, input_, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=None, name="deconv2d", spectral_normed=False, reuse=False, padding="SAME"):
        # Glorot initialization
        # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
        fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
        fan_out = k_h * k_w * output_shape[-1]
        if stddev is None:
            stddev = np.sqrt(2. / (fan_in))

        with tf.variable_scope(name, reuse=reuse) as scope:
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            if spectral_normed:
                deconv = tf.nn.conv2d_transpose(input_, self.spectral_norm(w),
                                            output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1], padding=padding)
            else:
                deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1], padding=padding)

            biases = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), tf.shape(deconv))
            
        return deconv

    def generator(self, z, reuse=False, spectral_normed=False):
        _batch_size = tf.shape(z)[0]
        with tf.variable_scope('G', reuse=reuse) as vs:
            net = self.linear(z, 7*7*128, name='fc_1', spectral_normed=spectral_normed, reuse=reuse)
            net = tf.nn.relu(net)
            net = tf.reshape(net, [_batch_size, 7, 7 , 128])
            net = self.deconv2d(net, [_batch_size, 14, 14, 64, ], name='deconv_1', spectral_normed=spectral_normed, reuse=reuse)
            net = tf.nn.relu(net)
            net = self.deconv2d(net, [_batch_size, 28, 28, 1], name='deconv_2', spectral_normed=spectral_normed, reuse=reuse)
            G_sample = tf.sigmoid(net)
        G_variables = tf.contrib.framework.get_variables(vs)
        return G_sample, G_variables
    
    def discriminator(self, X, spectral_normed=False, reuse=False):
        X = tf.reshape(X, [-1, 28, 28, 1])
        _batch_size = tf.shape(X)[0]
        with tf.variable_scope('D', reuse=reuse) as vs:
            net = self.conv2d(X, 128, name='conv_1', spectral_normed=spectral_normed, reuse=reuse)
            net = tf.nn.relu(net)# net = tf.nn.leaky_relu(net)
            net = self.conv2d(net, 64, name='conv_2', spectral_normed=spectral_normed, reuse=reuse)
            net = tf.nn.relu(net)# net = tf.nn.leaky_relu(net)
            net = tf.reshape(net, [-1, 7*7*64])
            D_logits = self.linear(net, 1, name='fc_1', spectral_normed=spectral_normed, reuse=reuse)
            D_logits = tf.sigmoid(D_logits)
        D_variables = tf.contrib.framework.get_variables(vs)
        return D_logits, D_variables

class LossDesign(object):

    def __init__(self):
        pass

    def gan_loss(self, D_real_logits, D_fake_logits, gan_type='GAN', relativistic=False):

        if relativistic:
            real_logits = (D_real_logits - tf.reduce_mean(D_fake_logits))
            fake_logits = (D_fake_logits - tf.reduce_mean(D_real_logits))
            if gan_type == 'GAN':
                D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits), logits=real_logits))
                D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits), logits=fake_logits))

                G_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(real_logits), logits=real_logits))
                G_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits), logits=fake_logits))
            
            else:
                raise NotImplementedError
    
        else:
            # Original GAN Loss design
            real_logits = D_real_logits
            fake_logits = D_fake_logits
            if gan_type == 'GAN':
                D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real_logits), logits=real_logits)) # Wish Discriminator give high score(close to 1) to the real data
                D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake_logits), logits=fake_logits)) # Wish Discriminator give low score(close to 0) to the fake data

                G_real_loss = 0
                G_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake_logits), logits=fake_logits)) # Wish Discriminator give high score(close to 1) to the fake data
            else:
                raise NotImplementedError
        
        D_loss = D_real_loss + D_fake_loss
        G_loss = G_real_loss + G_fake_loss

        return D_loss, G_loss