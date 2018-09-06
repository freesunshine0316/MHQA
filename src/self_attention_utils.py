import tensorflow as tf
import math
import layer_utils


initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale=3e-7)


def calcualte_absolute_positional_embedding(x, channels=None, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    inshape = tf.shape(x)
    batch_size = inshape[0]
    length = inshape[1]
    if channels is None:
        channels = inshape[2]

    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    signal = tf.tile(signal, multiples=[batch_size, 1, 1])  # [batch_size, length, channels]
    return signal


def layer_norm(inputs, epsilon=1e-6, dtype=None, scope=None, reuse=False):
    """
    Layer Normalization
    :param inputs: A Tensor of shape [..., channel_size]
    :param epsilon: A floating number
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string
    :returns: A Tensor with the same shape as inputs
    """
    with tf.variable_scope(scope, default_name="layer_norm", values=[inputs], dtype=dtype, reuse=reuse):
        channel_size = inputs.get_shape().as_list()[-1]
        scale = tf.get_variable("scale", shape=[channel_size], initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", shape=[channel_size], initializer=tf.zeros_initializer())
        mean = tf.reduce_mean(inputs, -1, True)
        variance = tf.reduce_mean(tf.square(inputs - mean), -1, True)
        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)
        return norm_inputs * scale + offset


def feedforward_layer(inputs, hidden_size, output_size, activation=tf.nn.relu,
                      scope_name=None, reuse=False, is_training=True, dropout_rate=0.2):
    with tf.variable_scope(scope_name, reuse=reuse):
        hiddens = tf.layers.dense(inputs, hidden_size, activation=activation, name='input_layer_' + scope_name)
        hiddens = layer_utils.dropout_layer(hiddens, dropout_rate, is_training=is_training)
        outputs = tf.layers.dense(hiddens, output_size, activation=activation, name='output_layer_' + scope_name)
    return outputs


def conv_block(inputs, num_conv_layers, kernel_size, num_filters,
               scope="conv_block", is_training=True,
               reuse=None, bias=True, dropout_rate=0.0):
    '''

    :param inputs: [batch_size, seq_length, word_dim]
    :param num_conv_layers: number of convolutional layers
    :param kernel_size:
    :param num_filters:
    :param scope:
    :param is_training:
    :param reuse:
    :param bias:
    :param dropout:
    :param sublayers:
    :return:
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.expand_dims(inputs,2) # [batch_size, seq_length, 'x', word_dim]
        for i in range(num_conv_layers):
            residual = outputs
            outputs = layer_norm(outputs, scope="layer_norm_%d"%i, reuse=reuse)
            if (i) % 2 == 0:
                outputs = layer_utils.dropout_layer(outputs, dropout_rate, is_training=is_training)
            outputs = depthwise_separable_convolution(outputs,
                kernel_size = (kernel_size, 1), num_filters=num_filters,
                scope="depthwise_conv_layers_%d"%i, is_training=is_training, reuse=reuse, bias=bias)
            outputs += residual
        return tf.squeeze(outputs,2)


def depthwise_separable_convolution(inputs, kernel_size, num_filters,
                                    scope="depthwise_separable_convolution",
                                    bias=True, is_training=True, reuse=None):
    '''

    :param inputs: [batch_size, seq_length, 'x', word_dim]
    :param kernel_size:
    :param num_filters:
    :param scope:
    :param bias:
    :param is_training:
    :param reuse:
    :return:
    '''
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()
        depthwise_filter = tf.get_variable("depthwise_filter",
                                        (kernel_size[0], kernel_size[1], shapes[-1], 1),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        pointwise_filter = tf.get_variable("pointwise_filter",
                                        (1,1,shapes[-1],num_filters),
                                        dtype = tf.float32,
                                        regularizer=regularizer,
                                        initializer = initializer_relu())
        outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter, strides = (1,1,1,1), padding = "SAME")
        if bias:
            b = tf.get_variable("bias",
                    outputs.shape[-1],
                    regularizer=regularizer,
                    initializer = tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs


def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])


def multihead_attention(inputs1, inputs2, hidden_size, head_size, mask1=None, mask2=None, remove_diagnoal=False, attn_type='dot',
                        scope="Multi_Head_Attention", reuse=None, is_training=True, dropout_rate=0.2, activation=tf.nn.relu):
    in_shapes = tf.shape(inputs1)
    batch_size = in_shapes[0]
    seq_len1 = in_shapes[1]
    seq_len2 = tf.shape(inputs2)[1]
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.layers.dense(inputs1, head_size * hidden_size, activation=activation, name='inputs1_' + scope)
        Q = tf.reshape(Q, [batch_size, seq_len1, head_size, hidden_size]) # [batch_size, seq_len1, head_size, hidden_size]
        Q = tf.transpose(Q, perm=[0, 2, 1, 3]) # [batch_size, head_size, seq_len1, hidden_size]
        Q = tf.reshape(Q, [batch_size*head_size, seq_len1,  hidden_size])

        K = tf.layers.dense(inputs2, head_size * hidden_size, activation=activation, name='inputs2_' + scope)
        K = tf.reshape(K, [batch_size, seq_len2, head_size, hidden_size]) # [batch_size, seq_len2, head_size, hidden_size]
        K = tf.transpose(K, perm=[0, 2, 1, 3]) # [batch_size, head_size, seq_len2, hidden_size]
        K = tf.reshape(K, [batch_size*head_size, seq_len2,  hidden_size])

        Q *= hidden_size**-0.5
        if mask1 is not None:
            mask1 = tf.reshape(mask1, shape=[batch_size, 1, seq_len1])  # [batch_size, 1, seq_len1]
            mask1 = tf.tile(mask1, multiples=[1, head_size, 1])  # [batch_size, head_size, seq_len1]
            mask1 = tf.reshape(mask1, shape=[batch_size*head_size, seq_len1])  # [batch_size*head_size, seq_len1]

        if mask2 is not None:
            mask2 = tf.reshape(mask2, shape=[batch_size, 1, seq_len2])  # [batch_size, 1, seq_len1]
            mask2 = tf.tile(mask2, multiples=[1, head_size, 1])  # [batch_size, head_size, seq_len1]
            mask2 = tf.reshape(mask2, shape=[batch_size*head_size, seq_len2])  # [batch_size*head_size, seq_len1]

        atten_scores = layer_utils.calcuate_attention(
            Q, K, hidden_size, hidden_size, scope_name='attention', att_type=attn_type,
            remove_diagnoal=remove_diagnoal, mask1=mask1, mask2=mask2, is_training=is_training,
            dropout_rate=dropout_rate) # [batch_size*head_size, seq_len1, seq_len2]

        atten_scores = tf.reshape(atten_scores, [batch_size, head_size, seq_len2]) # [batch_size, head_size, seq_len2]
        atten_scores = tf.transpose(atten_scores, [0, 2, 1]) # [batch_size, seq_len2, head_size]
        atten_scores = tf.layers.dense(atten_scores, 1, activation=activation, name='final_projection') # [batch_size, seq_len2, 1]
        atten_scores = tf.reshape(atten_scores, [batch_size, seq_len2]) # [batch_size, seq_len2]
        return atten_scores


###############################


def basic_block(inputs, in_dim, mask=None, scope="basic_block", is_training=True, reuse=None, options=None,activation=tf.nn.relu,
                with_conv_block=False, add_position_per_step=False):
    with tf.variable_scope(scope, reuse=reuse):
        # add positional embedding
        hiddens = inputs
        if add_position_per_step:
            hiddens += calcualte_absolute_positional_embedding(hiddens)

        # convolutional
        if with_conv_block:
            hiddens = conv_block(hiddens, options.num_conv_layers, options.kernel_size, in_dim,
                       scope="conv_block", is_training=is_training, reuse=reuse, bias=True, dropout_rate=options.dropout_rate)

        # self attention
        hiddens_self_match = layer_norm(hiddens, scope="layer_norm_for_self_attention", reuse=reuse)
        (hiddens_self_match, self_match_dim) = multihead_attention(
            hiddens_self_match, hiddens_self_match, options.hidden_size, options.head_size, mask1=mask, mask2=mask, remove_diagnoal=True,
            scope="Multi_Head_Attention", reuse=reuse, is_training=is_training, dropout_rate=options.dropout_rate, activation=activation)

        outputs = tf.concat(axis=2, values=[hiddens, hiddens_self_match])
        out_dim = in_dim + self_match_dim

        # feedforward layer
        resedual = outputs
        outputs = layer_norm(outputs, scope="layer_norm_for_ffn", reuse=reuse)
        outputs = feedforward_layer(outputs, options.hidden_size, out_dim, activation=activation,
                              scope_name="ffn", reuse=reuse, is_training=is_training, dropout_rate=options.dropout_rate)
        outputs += resedual
        return (outputs, out_dim)


def Localized_match(passage, question, passage_mask, question_mask, onehot_binary=None,
               options=None, scope_name='Localized_match_layer2', is_training=True, reuse=False, activation=tf.nn.relu):
    with tf.variable_scope(scope_name, reuse=reuse):

        match_result = []
        match_dim = 0
        for i in xrange(options.match_layer_size):
            with tf.variable_scope(scope_name + "_{}".format(i), reuse=reuse):
                # compress input
                passage = tf.layers.dense(passage, options.hidden_size, activation=activation, name='input_layer_' + scope_name)
                # passage = feedforward_layer(passage, options.hidden_size, options.hidden_size, activation=activation,
                #           scope_name="compress_input", reuse=False, is_training=is_training, dropout_rate=options.dropout_rate)
                question = tf.layers.dense(question, options.hidden_size, activation=activation, name='input_layer_' + scope_name, reuse=True)
                # question = feedforward_layer(question, options.hidden_size, options.hidden_size, activation=activation,
                #                     scope_name="compress_input", reuse=True, is_training=is_training, dropout_rate=options.dropout_rate)
                # encoding layer
                (passage, out_dim) = basic_block(passage, options.hidden_size, mask=passage_mask, scope="encoding_block",
                              is_training=is_training, reuse=None, options=options,activation=activation,
                                         with_conv_block=options.with_conv_block,
                                         add_position_per_step=options.add_position_per_step)

                (question, out_dim) = basic_block(question, options.hidden_size, mask=question_mask, scope="encoding_block",
                              is_training=is_training, reuse=True, options=options, activation=activation,
                                          with_conv_block=options.with_conv_block,
                                          add_position_per_step=options.add_position_per_step)

                # matching layer
                (passage_match, passage_match_dim) = multihead_attention(
                    passage, question, options.hidden_size, options.head_size, mask1=passage_mask, mask2=question_mask,
                    remove_diagnoal=False, scope="passage_match", reuse=None, is_training=is_training,
                    dropout_rate=options.dropout_rate, activation=activation)

                if i < options.match_layer_size - 1:
                    (question_match, question_match_dim) = multihead_attention(
                        question, passage, options.hidden_size, options.head_size, mask2=passage_mask, mask1=question_mask,
                        remove_diagnoal=False, scope="question_match", reuse=None, is_training=is_training,
                        dropout_rate=options.dropout_rate, activation=activation)
                    question = tf.concat(axis=2, values=[question, question_match])

                passage = tf.concat(axis=2, values=[passage, passage_match])
                match_result.append(passage)
                match_dim += passage_match_dim + out_dim

        match_result = tf.concat(axis=2, values=match_result)
        (match_result, out_dim) = basic_block(match_result, match_dim, mask=passage_mask, scope="match_block",
                                         is_training=is_training, reuse=None, options=options, activation=activation)
        return match_result, out_dim


