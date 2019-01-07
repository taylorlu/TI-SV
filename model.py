# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import numpy as np
import random
import tensorflow as tf
from modules import *
import tensorflow.contrib.slim as slim

class AudioMeta():

    def __init__(self, data_path):
        self.cur_index = 0
        self.np_file_list = os.listdir(data_path)
        inputs = []
        labels = []
        for (i, file) in enumerate(self.np_file_list):
            utters = np.load(os.path.join(data_path, file)).transpose((0,2,1))  # (n, t, d)
            # print("{}, utters.shape = {}".format(file, utters.shape))
            spk_input = np.split(utters, utters.shape[0], axis=0)
            spk_label = [i]*utters.shape[0]
            # print("len(spk_input) = {}, len(spk_label) = {}".format(len(spk_input), len(spk_label)))
            inputs.extend(spk_input)
            labels.extend(spk_label)
        # print("len(inputs) = {}, len(labels) = {}".format(len(inputs), len(labels)))
        # print("{}, {}".format(inputs[10000].shape, labels[10000]))
        self.data_list = list(zip(inputs, labels))
        random.shuffle(self.data_list)
        self.inputs, self.labels = zip(*self.data_list)

    @property
    def num_speakers(self):
        return len(self.np_file_list)

    @property
    def num_utters(self):
        return len(self.data_list)

    def get_batch(self, batch_size=100):
        self.cur_index += batch_size
        if(self.cur_index>self.num_utters):
            random.shuffle(self.data_list)
            self.inputs, self.labels = zip(*self.data_list)
            self.cur_index = batch_size
        s_slice = np.random.randint(0, 16)
        e_slice = np.random.randint(140, 156)
        batch_input = np.array(self.inputs[self.cur_index-batch_size: self.cur_index]).squeeze()
        batch_input = batch_input[:, s_slice:e_slice, :]
        batch_label = np.array(self.labels[self.cur_index-batch_size: self.cur_index]).squeeze()
        return batch_input, batch_label


class ClassificationModel():
    '''
    n = batch size
    t = timestep size
    h = hidden size
    e = embedding size
    '''
    def __init__(self, num_banks=8, num_clusters=32, hidden_units=128, num_highway=2, norm_type='ins', num_classes=100, bottleneck_size=256):
        self.num_banks = num_banks
        self.num_clusters = num_clusters
        self.hidden_units = hidden_units
        self.num_highway = num_highway
        self.norm_type = norm_type
        self.num_classes = num_classes
        self.bottleneck_size = bottleneck_size
        self.center_loss_factor = 0.6
        self.norm_loss_factor = 0.6
        self.audioMeta = AudioMeta('/home/logview/workspace/projects/Speaker_Verification/train_tisv')
        self.melInputs = tf.placeholder(tf.float32, [None, None, 40], name='mels')
        self.labelInputs = tf.placeholder(tf.int32, [None], name='labels')

    def getBatch_data_label(self, batch_size=100):
        batch_input, batch_label = self.audioMeta.get_batch()
        return batch_input, batch_label

    def embedding(self, x, is_training=True):
        """
        :param x: shape=(n, t, n_mels)
        :return: embedding. shape=(n, e)
        """
        # frame-level embedding
        x = conv1d(x, self.hidden_units, 1, scope="conv1d") # (n, t, h)

        out = conv1d_banks(x, K=self.num_banks, num_units=self.hidden_units, norm_type=self.norm_type,
                           is_training=is_training)  # (n, t, k * h)

        out = tf.layers.max_pooling1d(out, 2, 1, padding="same")  # (n, t, k * h)

        out = conv1d(out, self.hidden_units, 3, scope="conv1d_1")  # (n, t, h)
        out = normalize(out, type=self.norm_type, is_training=is_training, activation_fn=tf.nn.relu)
        out = conv1d(out, self.hidden_units, 3, scope="conv1d_2")  # (n, t, h)
        out += x  # (n, t, h) # residual connections

        for i in range(self.num_highway):
            out = highwaynet(out, num_units=self.hidden_units, scope='highwaynet_{}'.format(i))  # (n, t, h)

        return out

    def margin_loss(self, features, label, centers, beta):
        #features need to do L2 norm before process since beta make sense

        batchSize = tf.shape(features)[0]
        val = centers - tf.reshape(features, [tf.shape(features)[0], 1, tf.shape(features)[1]])
        distance = tf.reduce_sum(tf.square(val), 2)
        var_distance = tf.Variable(0,name='temp', dtype=distance.dtype)
        var_distance = tf.assign(var_distance, distance, validate_shape=False)

        seq = tf.range(batchSize)
        zipper = tf.stack([seq, label], 1)
        c_distance = tf.gather_nd(distance, zipper)    #change the value of batch's own center to MAX_FLOAT
        var_distance = tf.scatter_nd_update(var_distance, zipper, tf.ones(batchSize, dtype=tf.float32)*np.finfo(np.float32).max)

        minIndexs = tf.cast(tf.argmin(var_distance, 1), tf.int32)
        minIndexs = tf.stack([seq, minIndexs], 1)
        minValue = tf.gather_nd(var_distance, minIndexs)    #calc minDistance between feature of whole centers(except its own center)

        basic_loss = tf.add(tf.subtract(c_distance, minValue), beta)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

        return loss

    def margin_center_loss(self, features, label, nrof_classes):

        nrof_features = features.get_shape()[1]
        centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
        label = tf.reshape(label, [-1])

        mloss = self.margin_loss(features, label, centers, 0.6)

        centers_batch = tf.gather(centers, label)
        diff = (1 - 0.6) * (centers_batch - features)
        centers = tf.scatter_sub(centers, label, diff)
        with tf.control_dependencies([centers]):
            loss = tf.reduce_mean(tf.square(features - centers_batch))

        return loss+mloss

    def totalLoss(self, bottleneck, label_batch):
        logits = slim.fully_connected(bottleneck, self.num_classes, activation_fn=None,
                weights_initializer=slim.initializers.xavier_initializer(),
                scope='Logits', reuse=False)

        # # Norm for the prelogits
        # eps = 1e-4
        norm_loss = tf.reduce_mean(tf.abs(tf.norm(bottleneck, axis=1)-tf.ones(tf.shape(bottleneck)[0], dtype=tf.float32))) * self.norm_loss_factor

        # Add center loss and margin loss
        margin_center_loss = self.margin_center_loss(bottleneck, label_batch, self.num_classes) * self.center_loss_factor

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        # # Calculate the total losses
        total_loss = cross_entropy_loss + margin_center_loss + norm_loss

        return total_loss

    def buildModel(self, is_training=True):
        with tf.variable_scope('embedding'):
            logits = self.embedding(self.melInputs, is_training=is_training)  # (n, t ,d)

        with tf.variable_scope("vlad"):
            inputs = tf.reshape(logits, [-1, logits.get_shape()[-1]]) #(n*t, d)
            # alfas = tf.get_variable('alfa', dtype=tf.float32, shape=[self.num_clusters]) #(k)
            alfas_pow = tf.ones(self.num_clusters)*100
            # alfas_pow = tf.square(alfas) +1e-10
            MUs = tf.get_variable("mu", shape=[self.num_clusters, logits.get_shape()[-1]], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

            weights = tf.matmul( 2*tf.matrix_diag(alfas_pow), MUs ) #(k, d)
            biases = -tf.multiply(alfas_pow, tf.reduce_sum(tf.square(MUs),1)) #(k)
            output = tf.matmul(inputs, tf.transpose(weights)) + biases #(n*t, k)
            PI = tf.nn.softmax(output) #(n*t, k)

            tileInputs = tf.tile(inputs, [1, self.num_clusters])
            tileInputs2 = tf.reshape(tileInputs, [-1, self.num_clusters, logits.get_shape()[-1]])    # (n*t, k, d)

            offcenter = tileInputs2 - MUs # (n*t, k, d)

            ####
            offcenter = tf.reshape(offcenter, [tf.shape(logits)[0], tf.shape(logits)[1], self.num_clusters, logits.get_shape()[-1]]) #(n,t,k,h)
            offcenter = tf.transpose(offcenter, [0, 2, 1, 3])    # (n, k, t, h)

            PI = tf.transpose(tf.reshape(PI, [tf.shape(logits)[0], tf.shape(logits)[1], self.num_clusters]), [0, 2, 1])  # (n, k, t)
            tilePI = tf.tile(tf.expand_dims(PI, -1), [1, 1, 1, logits.get_shape()[-1]]) # (n, k, t, h)
            offcenter = tf.multiply(tilePI, offcenter)    # (n, k, t, h)
            vladVector = tf.reduce_sum(offcenter, 2)    # (n, k, h)
            vladVector = tf.reshape(vladVector, [-1, self.num_clusters*logits.get_shape()[-1]])    # (n, k*h)
            print(vladVector.get_shape())

        # fc to bottleneck
        bottleneck = tf.layers.dense(vladVector, self.bottleneck_size, name='projection')  # (n, e)

        normVector = tf.nn.l2_normalize(bottleneck, 1, 1e-10, name='normVector')

        totalloss = self.totalLoss(bottleneck, self.labelInputs)

        return totalloss, normVector, tf.reduce_mean(alfas_pow)
