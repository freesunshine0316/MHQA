
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from self_attention_utils import multihead_attention
from layer_utils import dropout_layer
import match_utils

def clip_and_normalize(word_probs, epsilon):
    word_probs = tf.clip_by_value(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / tf.reduce_sum(word_probs, axis=-1, keep_dims=True)


# memory [batch, entity, feat_dim]
# query [batch, q_len, feat_dim]
def multi_perspective_logits(memory, memory_len, memory_mask,
        query, query_len, query_mask, options, is_training):
    feat_dim = options.seq_lstm_dim * 2
    match_rep, match_dim = match_utils.multi_granularity_match(feat_dim, memory, query, memory_len, query_len,
            passage_mask=memory_mask, question_mask=query_mask, is_training=is_training, dropout_rate=options.dropout_rate,
            options=options, with_full_matching=False, with_attentive_matching=True,
            with_max_attentive_matching=True)
    # match_rep [batch, entity, match_dim]
    proj_w = tf.get_variable('proj_w', [match_dim, 1], dtype=tf.float32)
    proj_b = tf.get_variable('proj_b', [1], dtype=tf.float32)
    batch_size = tf.shape(match_rep)[0]
    entity_num = tf.shape(match_rep)[1]
    print(match_dim)
    logits = tf.reshape(match_rep, [batch_size*entity_num, match_dim])
    logits = tf.matmul(logits, proj_w) + proj_b
    logits = tf.reshape(logits, [batch_size, entity_num])
    return logits #nn_ops.softmax(logits) * memory_mask


def attention_logits(in_value_1, in_value_2, feature_dim1, feature_dim2, mask1, mask2,
        options, is_training=False, scope_name='attention'):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, options.dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, options.dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, options.attn_depth], dtype=tf.float32)
        atten_value_1 = tf.matmul(tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)  # [batch_size*len_1, feature_dim]

        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, options.attn_depth], dtype=tf.float32)
        atten_value_2 = tf.matmul(tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)  # [batch_size*len_2, feature_dim]

        if options.attn_type == 'additive':
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, options.attn_depth])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, options.attn_depth])

            atten_b = tf.get_variable("atten_b", [options.attn_depth], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, options.attn_depth], dtype=tf.float32)
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")  # [batch_size, len_1, 'x', feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")  # [batch_size, 'x', len_2, feature_dim]
            atten_value = atten_value_1 + atten_value_2
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            atten_value = tf.tanh(atten_value)  # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, options.attn_depth]) * atten_v  # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        elif options.attn_type == 'symmetric':
            atten_value_1 = tf.nn.relu(atten_value_1) # [batch_size*len1, att_dim]
            atten_value_2 = tf.nn.relu(atten_value_2) # [batch_size*len2, att_dim]
            D_in = tf.get_variable("diagonal_{}".format(scope_name), [options.attn_depth], dtype=tf.float32)  # att_dim
            D = D_in * tf.diag(tf.ones([options.attn_depth], tf.float32), name='diagonal')  # att_dim xatt_dim
            atten_value_1 = tf.matmul(atten_value_1, D)  # [batch_size*len1, att_dim]
            atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, options.attn_depth])
            atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, options.attn_depth])
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True) # [batch_size, len_1, len_2]
        else:
            assert False, 'unsupported type'

    return atten_value


def collect_node(representation, positions):
    # representation: [batch_size, num_nodes, feature_dim]
    # positions: [batch_size, num_poses]
    batch_size = tf.shape(positions)[0]
    node_num = tf.shape(positions)[1]
    rids = tf.range(0, limit=batch_size) # [batch,]
    rids = tf.reshape(rids, [-1, 1]) # [batch, 1,]
    rids = tf.tile(rids, [1, node_num]) # [batch, num_poses,]
    indices = tf.stack((rids, positions), axis=2) # [batch, num_poses, 2]
    return tf.gather_nd(representation, indices) # [batch, num_poses, feature_dim]


def collect_neighbor(representation, positions):
    # representation: [batch_size, num_nodes, feature_dim]
    # positions: [batch_size, num_nodes, num_neighbors]
    feature_dim = tf.shape(representation)[2]
    input_shape = tf.shape(positions)
    batch_size = input_shape[0]
    num_nodes = input_shape[1]
    num_neighbors = input_shape[2]
    positions_flat = tf.reshape(positions, [batch_size, num_nodes*num_neighbors])
    def singel_instance(x):
        # x[0]: [num_nodes, feature_dim]
        # x[1]: [num_nodes*num_neighbors]
        return tf.gather(x[0], x[1])
    elems = (representation, positions_flat)
    representations = tf.map_fn(singel_instance, elems, dtype=tf.float32)
    return tf.reshape(representations, [batch_size, num_nodes, num_neighbors, feature_dim])


def collect_neighbor_v2(representation, positions):
    # representation: [batch_size, num_nodes, feature_dim]
    # positions: [batch_size, num_nodes, num_neighbors]
    batch_size = tf.shape(positions)[0]
    node_num = tf.shape(positions)[1]
    neigh_num = tf.shape(positions)[2]
    rids = tf.range(0, limit=batch_size) # [batch]
    rids = tf.reshape(rids, [-1, 1, 1]) # [batch, 1, 1]
    rids = tf.tile(rids, [1, node_num, neigh_num]) # [batch, nodes, neighbors]
    indices = tf.stack((rids, positions), axis=3) # [batch, nodes, neighbors, 2]
    return tf.gather_nd(representation, indices) # [batch, nodes, neighbors, feature_dim]


def collect_final_step(lstm_rep, lens):
    lens = tf.maximum(lens, tf.zeros_like(lens, dtype=tf.int32)) # [batch,]
    idxs = tf.range(0, limit=tf.shape(lens)[0]) # [batch,]
    indices = tf.stack((idxs,lens,), axis=1) # [batch_size, 2]
    return tf.gather_nd(lstm_rep, indices, name='lstm-forward-last')

