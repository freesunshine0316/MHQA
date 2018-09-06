import tensorflow as tf
from tensorflow.python.ops import nn_ops

def my_lstm_layer(input_reps, lstm_dim, scope_name=None, reuse=False, is_training=True, dropout_rate=0.2):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param lstm_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    with tf.variable_scope(scope_name, reuse=reuse):
        inputs = tf.transpose(input_reps, [1, 0, 2])
        inputs = dropout_layer(inputs, dropout_rate, is_training=is_training)
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(1, lstm_dim, direction="bidirectional",
                                    name="{}_cudnn_bi_lstm".format(scope_name), dropout=0)
                                    # name="{}_cudnn_bi_lstm".format(scope_name), dropout=dropout_rate if is_training else 0)
        outputs, _ = lstm(inputs)
    outputs = tf.transpose(outputs, [1, 0, 2])
    f_rep = outputs[:, :, 0:lstm_dim]
    b_rep = outputs[:, :, lstm_dim:2*lstm_dim]
    return (f_rep,b_rep, outputs)

def dropout_layer(input_reps, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr

def cosine_distance(y1,y2, cosine_norm=True, eps=1e-6):
    # cosine_norm = True
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm

def euclidean_distance(y1, y2, eps=1e-6):
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    return distance

def softmax_with_mask(values, mask):
    # e_x = np.exp(x - np.max(x))
    # return e_x / e_x.sum()
    e_x = tf.exp(values - tf.expand_dims(tf.reduce_max(values, axis=-1), axis=-1))
    # e_x *= tf.expand_dims(mask, axis=-1)
    e_x += 1e-6
    e_x *= mask
    return e_x / (tf.expand_dims(tf.reduce_sum(e_x, axis=-1), axis=-1) + 1e-6)

def cross_entropy(logits, truth, mask=None):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]
    if mask is not None: logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
    result = tf.multiply(truth, log_predictions) # [batch_size, passage_len]
    if mask is not None: result = tf.multiply(result, mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]

def projection_layer(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
    with tf.variable_scope(scope or "projection_layer"):
        full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        outputs = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs # [batch_size, passage_len, output_size]

def highway_layer(in_val, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, activation_func=tf.tanh, scope_name=None, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        for i in xrange(num_layers):
            cur_scope_name = scope_name + "-{}".format(i)
            in_val = highway_layer(in_val, output_size,activation_func=activation_func, scope=cur_scope_name)
    return in_val

def collect_representation2(representation, positions):
    '''
    :param representation: [batch_size, passsage_length, dim]
    :param positions: [batch_size, num_positions]
    :return:
    '''
    def singel_instance(x):
        # x[0]: [passage_length, dim]
        # x[1]: [num_positions]
        return tf.gather(x[0], x[1])
    elems = (representation, positions)
    return tf.map_fn(singel_instance, elems, dtype=tf.float32) # [batch_size, num_positions, dim]

def collect_representation(representation, positions):
    # representation: [batch_size, node_num, feature_dim]
    # positions: [batch_size, neigh_num]
    return collect_probs(representation, positions)


def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
    return result # [batch_size, dim]

def collect_probs(probs, positions):
    # probs [batch_size, chunks_size]
    # positions [batch_size, pair_size]
    batch_size = tf.shape(probs)[0]
    pair_size = tf.shape(positions)[1]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    batch_nums = tf.reshape(batch_nums, shape=[-1, 1]) # [batch_size, 1]
    batch_nums = tf.tile(batch_nums, multiples=[1, pair_size]) # [batch_size, pair_size]

    indices = tf.stack((batch_nums, positions), axis=2) # shape (batch_size, pair_size, 2)
    pair_probs = tf.gather_nd(probs, indices)
    # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
    return pair_probs

def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None,
                       is_training=False, dropout_rate=0.2, cosine_attention_scale=200):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        if att_type != 'cosine' and att_type != 'dot':
            # calculate attention ==> a: [batch_size, len_1, len_2]
            atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]

            if feature_dim1 == feature_dim2: atten_w2 = atten_w1
            else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
            # atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
            atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]

        if att_type == 'additive':
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])

            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        elif att_type == 'symmetric':
            atten_value_1 = tf.nn.relu(atten_value_1) # [batch_size*len1, att_dim]
            atten_value_2 = tf.nn.relu(atten_value_2) # [batch_size*len2, att_dim]
            D_in = tf.get_variable("diagonal_{}".format(scope_name), [att_dim], dtype=tf.float32)  # att_dim
            D = D_in * tf.diag(tf.ones([att_dim], tf.float32), name='diagonal')  # att_dim xatt_dim
            atten_value_1 = tf.matmul(atten_value_1, D)  # [batch_size*len1, att_dim]
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]
        elif att_type == 'cosine':
            atten_value = cal_relevancy_matrix(in_value_2, in_value_1)
            atten_value = atten_value * cosine_attention_scale
        elif att_type == 'dot':
            atten_value = tf.matmul(in_value_1, in_value_2, transpose_b=True)
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagonal_params", [att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * tf.expand_dims(diagnoal_params, axis=0)
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]

        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        # normalize
        # atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        atten_value = softmax_with_mask(atten_value, tf.expand_dims(mask2, axis=1))
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value


def calcuate_attention_bak(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None, is_training=False, dropout_rate=0.2,
                       cosine_attention_scale=200):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # if att_type != 'cosine':
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]

        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        # atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]

        if att_type == 'additive':
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])

            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2  # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        elif att_type == 'symmetric':
            atten_value_1 = tf.nn.relu(atten_value_1) # [batch_size*len1, att_dim]
            atten_value_2 = tf.nn.relu(atten_value_2) # [batch_size*len2, att_dim]
            D_in = tf.get_variable("diagonal_{}".format(scope_name), [att_dim], dtype=tf.float32)  # att_dim
            D = D_in * tf.diag(tf.ones([att_dim], tf.float32), name='diagonal')  # att_dim xatt_dim
            atten_value_1 = tf.matmul(atten_value_1, D)  # [batch_size*len1, att_dim]
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]
        elif att_type == 'cosine':
            atten_value_1 = tf.nn.relu(atten_value_1) # [batch_size*len1, att_dim]
            atten_value_2 = tf.nn.relu(atten_value_2) # [batch_size*len2, att_dim]
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])
            atten_value = cal_relevancy_matrix(atten_value_2, atten_value_1)
            atten_value = atten_value * cosine_attention_scale
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagonal_params", [att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * tf.expand_dims(diagnoal_params, axis=0)
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]

        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        # normalize
        # atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        atten_value = softmax_with_mask(atten_value, tf.expand_dims(mask2, axis=1))
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))

    return atten_value

def fusion_attention_amit(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=250, remove_diagnoal=False, mask1=None, mask2=None, is_training=False, dropout_rate=0.2):
# def fusion_attention_amit(scope_name, in_q_rep, in_p_rep,  w_dim1, w_dim2,
#                      reuse=False, is_training=None, options=None, remove_diagnoal=False):
    input_shape = tf.shape(in_value_2)
    batch_size = input_shape[0]
    question_len = input_shape[1]
    passage_len = tf.shape(in_value_1)[1]
    with tf.variable_scope(scope_name, reuse=False):
        q_rep = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
        p_rep = dropout_layer(in_value_1, dropout_rate, is_training=is_training)

        w_wordlevel = tf.get_variable("famf_highlevel_{}".format(scope_name), [feature_dim1, att_dim],
                                      dtype=tf.float32)  # D1 x D2
        q_rep = tf.reshape(q_rep, [-1, feature_dim1], name='BQ_dim1')  # [B * Q, D1]

        q_rep = tf.matmul(q_rep, w_wordlevel)  # [B * Q, D2]
        q_rep = tf.nn.relu(q_rep)  # [B * Q, D2]

        D_in = tf.get_variable("diagonal_{}".format(scope_name), [att_dim],
                                              dtype=tf.float32)  # D1 x D2
        D = D_in * tf.diag(tf.ones([att_dim], tf.float32), name='diagonal')  # D2 x D2
        q_rep = tf.matmul(q_rep, D)  # f(Ux) D  # [B * Q, D2]

        q_rep = tf.reshape(q_rep, [batch_size, question_len, att_dim],name='B_Q_dim2')  # [B, Q, D2]


        p_rep = tf.reshape(p_rep, [-1, feature_dim1],name='BP_dim1')  # [B * P, D1]
        p_rep = tf.matmul(p_rep, w_wordlevel)  # [B * P, D2]
        p_rep = tf.nn.relu(p_rep)  # f(Uy) # [B * P, D2]
        p_rep = tf.reshape(p_rep, [batch_size, passage_len,att_dim],name='B_P_dim2')  # [B, P, D2]

        # passage: B x P x D2
        # question: B x Q x D2
        shuffled = tf.transpose(q_rep, perm=[0, 2, 1])  # B x D2 x Q

        # similarity between passage and query
        S_q_p = tf.matmul(p_rep, shuffled)  # B x P x Q

        alphas = tf.nn.softmax(S_q_p)  # B x P x Q
        return alphas

        # expanded_alphas = tf.expand_dims(alphas, axis=-1)  # [ B , P , Q , 'x']
        # weighted_query = tf.expand_dims(query_rep, axis=1)  # [B, 'x', Q, D1]
        # weighted_query = tf.reduce_sum(tf.multiply(weighted_query, expanded_alphas), axis=2)  # [B, P, D1]
        # return weighted_query


def weighted_sum(atten_scores, in_values):
    '''

    :param atten_scores: # [batch_size, len1, len2]
    :param in_values: [batch_size, len2, dim]
    :return:
    '''
    return tf.matmul(atten_scores, in_values)

def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1) # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2) # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance(in_question_repres_tmp,in_passage_repres_tmp) # [batch_size, passage_len, question_len]
    return relevancy_matrix

def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    if question_mask is not None:
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix

def compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]
