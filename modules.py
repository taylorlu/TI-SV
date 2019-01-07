# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf


def normalize(inputs,
              type="bn",
              decay=.999,
              epsilon=1e-8,
              is_training=True,
              reuse=None,
              activation_fn=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension. Or if type is `ln`, the normalization is over
        the last dimension. Note that this is different from the native
        `tf.contrib.layers.batch_norm`. For this I recommend you change
        a line in ``tensorflow/contrib/layers/python/layers/layer.py`
        as follows.
        Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
        After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)
      type: A string. Either "bn" or "ln".
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance.
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type=="bn":
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims

        # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
        # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
        if inputs_rank in [2, 3, 4]:
            if inputs_rank==2:
                inputs = tf.expand_dims(inputs, axis=1)
                inputs = tf.expand_dims(inputs, axis=2)
            elif inputs_rank==3:
                inputs = tf.expand_dims(inputs, axis=1)

            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               decay=decay,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               zero_debias_moving_mean=True,
                                               fused=True,
                                               reuse=reuse)
            # restore original shape
            if inputs_rank==2:
                outputs = tf.squeeze(outputs, axis=[1, 2])
            elif inputs_rank==3:
                outputs = tf.squeeze(outputs, axis=1)
        else: # fallback to naive batch norm
            outputs = tf.contrib.layers.batch_norm(inputs=inputs,
                                               decay=decay,
                                               center=True,
                                               scale=True,
                                               updates_collections=None,
                                               is_training=is_training,
                                               scope=scope,
                                               reuse=reuse,
                                               fused=False)
    elif type in ("ln",  "ins"):
        reduction_axis = -1 if type=="ln" else 1
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [reduction_axis], keep_dims=True)
            beta = tf.get_variable("beta", shape=params_shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable("gamma", shape=params_shape, initializer=tf.ones_initializer)
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta
    else:
        outputs = inputs

    if activation_fn:
        outputs = activation_fn(outputs)

    return outputs


def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           use_bias=False,
           activation_fn=None,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower()=="causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":activation_fn,
                "use_bias":use_bias, "reuse":reuse}

        outputs = tf.layers.conv1d(**params)
    return outputs


def conv1d_banks(inputs, K=16, num_units=None, norm_type=None, is_training=True, scope="conv1d_banks", reuse=None):
    '''Applies a series of conv1d separately.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.

    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = []
        for k in range(1, K+1):
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, num_units, k)
                output = normalize(output, type=norm_type, is_training=is_training, activation_fn=tf.nn.relu)
            outputs.append(output)
        outputs = tf.concat(outputs, -1)
    return outputs # (N, T, Hp.embed_size//2*K)


def gru(inputs, num_units=None, bidirection=False, seqlens=None, scope="gru", reuse=None):
    '''Applies a GRU.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results
        are concatenated.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        # cell = tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(num_units,kernel_initializer=tf.orthogonal_initializer()),output_keep_prob=0.5)
        cell = tf.contrib.rnn.GRUCell(num_units,kernel_initializer=tf.orthogonal_initializer())
        if bidirection:
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs,
                                                         sequence_length=seqlens,
                                                         dtype=tf.float32)
            return tf.concat(outputs, 2)
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs,
                                           sequence_length=seqlens,
                                           dtype=tf.float32)
            return outputs


def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387

    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        C = 1. - T
        outputs = H * T + inputs * C
    return outputs
