
import tensorflow as tf
import match_utils
import operation_utils

class GCNEncoder(object):
    def __init__(self, entity_repre, entity_mask, entity_dim, edges, edges_mask,
            is_training=True, options=None):
        assert options != None

        # shapes
        batch_size = tf.shape(edges)[0]
        entity_size_max = tf.shape(edges)[1]

        with tf.variable_scope('gcn_encoder'):
            # transform entity_repre into entity_hidden
            mapping_w = tf.get_variable("mapping_w",
                    [entity_dim, options.grn_dim], dtype=tf.float32)
            mapping_b = tf.get_variable("mapping_b",
                    [options.grn_dim], dtype=tf.float32)

            entity_hidden = tf.reshape(entity_repre,
                    [-1, entity_dim])
            entity_hidden = tf.matmul(entity_hidden, mapping_w) + mapping_b
            entity_hidden = tf.reshape(entity_hidden,
                    [batch_size, entity_size_max, options.grn_dim])
            entity_hidden = entity_hidden * tf.expand_dims(entity_mask, axis=-1)

            # calculate graph representation
            w_ingate = tf.get_variable("w_ingate",
                    [options.grn_dim, options.grn_dim], dtype=tf.float32)
            b_ingate = tf.get_variable("b_ingate",
                    [options.grn_dim], dtype=tf.float32)

            self.grn_historys = []
            for i in xrange(options.num_grn_step):
                # [batch, entity, neighbor, grn_dim]
                neigh_prev_hidden = operation_utils.collect_neighbor_v2(entity_hidden, edges)
                neigh_prev_hidden = tf.multiply(neigh_prev_hidden,
                        tf.expand_dims(edges_mask, axis=-1))
                # [batch, entity, grn_dim]
                neigh_prev_hidden = tf.reduce_sum(neigh_prev_hidden, axis=2)
                neigh_prev_hidden = tf.multiply(neigh_prev_hidden,
                        tf.expand_dims(entity_mask, axis=-1))
                # [batch*entity, grn_dim]
                neigh_prev_hidden = tf.reshape(neigh_prev_hidden,
                        [-1, options.grn_dim])

                neigh_prev_hidden = tf.matmul(neigh_prev_hidden, w_ingate) + b_ingate
                neigh_prev_hidden = tf.sigmoid(neigh_prev_hidden)
                neigh_prev_hidden = tf.reshape(neigh_prev_hidden,
                        [batch_size, entity_size_max, options.grn_dim])
                entity_hidden = neigh_prev_hidden * tf.expand_dims(entity_mask, axis=-1)

                if is_training:
                    self.grn_historys.append(tf.nn.dropout(entity_hidden, (1 - options.dropout_rate)))
                else:
                    self.grn_historys.append(entity_hidden)

            # decide how to use graph_representations
            self.grn_hiddens = self.grn_historys[-1]

