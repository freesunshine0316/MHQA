
import tensorflow as tf
import match_utils
import operation_utils

class GraphEncoder(object):
    def __init__(self, entity_repre, entity_mask, entity_dim, edges, edges_mask,
            is_training=True, options=None):
        assert options != None

        # shapes
        batch_size = tf.shape(edges)[0]
        entity_size_max = tf.shape(edges)[1]

        with tf.variable_scope('graph_encoder'):
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

            entity_cell = tf.zeros([batch_size, entity_size_max, options.grn_dim],
                    dtype=tf.float32)

            u_ingate = tf.get_variable("u_ingate",
                    [options.grn_dim, options.grn_dim], dtype=tf.float32)
            b_ingate = tf.get_variable("b_ingate",
                    [options.grn_dim], dtype=tf.float32)

            u_forgetgate = tf.get_variable("u_forgetgate",
                    [options.grn_dim, options.grn_dim], dtype=tf.float32)
            b_forgetgate = tf.get_variable("b_forgetgate",
                    [options.grn_dim], dtype=tf.float32)

            u_outgate = tf.get_variable("u_outgate",
                    [options.grn_dim, options.grn_dim], dtype=tf.float32)
            b_outgate = tf.get_variable("b_outgate",
                    [options.grn_dim], dtype=tf.float32)

            u_cell = tf.get_variable("u_cell",
                    [options.grn_dim, options.grn_dim], dtype=tf.float32)
            b_cell = tf.get_variable("b_cell",
                    [options.grn_dim], dtype=tf.float32)

            # calculate question graph representation
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

                ## ig
                ingate = tf.sigmoid(tf.matmul(neigh_prev_hidden, u_ingate) + b_ingate)
                ingate = tf.reshape(ingate, [batch_size, entity_size_max, options.grn_dim])
                ## fg
                forgetgate = tf.sigmoid(tf.matmul(neigh_prev_hidden, u_forgetgate) + b_forgetgate)
                forgetgate = tf.reshape(forgetgate, [batch_size, entity_size_max, options.grn_dim])
                ## og
                outgate = tf.sigmoid(tf.matmul(neigh_prev_hidden, u_outgate) + b_outgate)
                outgate = tf.reshape(outgate, [batch_size, entity_size_max, options.grn_dim])
                ## input
                cell_input = tf.tanh(tf.matmul(neigh_prev_hidden, u_cell) + b_cell)
                cell_input = tf.reshape(cell_input, [batch_size, entity_size_max, options.grn_dim])
                ## gated operation
                new_entity_cell = forgetgate * entity_cell + ingate * cell_input
                new_entity_hidden = outgate * tf.tanh(new_entity_cell)
                # apply mask
                entity_cell = tf.multiply(new_entity_cell, tf.expand_dims(entity_mask, axis=-1))
                entity_hidden = tf.multiply(new_entity_hidden, tf.expand_dims(entity_mask, axis=-1))

                if is_training:
                    self.grn_historys.append(tf.nn.dropout(entity_hidden, (1 - options.dropout_rate)))
                else:
                    self.grn_historys.append(entity_hidden)

            # decide how to use graph_representations
            self.grn_hiddens = self.grn_historys[-1]
            self.grn_cells = entity_cell

