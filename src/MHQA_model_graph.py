
import tensorflow as tf
from tensorflow.python.ops import nn_ops
import numpy as np

import encoder_utils
import graph_encoder_utils
import gcn_encoder_utils
import padding_utils
import operation_utils

import random


class ModelGraph(object):
    def __init__(self, word_vocab, char_vocab=None, POS_vocab=None, NER_vocab=None, options=None, \
            has_ref=True, is_training=True):
        # is_training controls whether to use dropout and update parameters
        self.is_training = is_training
        # has_ref distinguish 'dev' evaluation from 'final test' evaluation
        self.has_ref = has_ref

        self.options = options
        self.word_vocab = word_vocab

        # separately encode passage and question
        self.passage_encoder = encoder_utils.SeqEncoder(options, word_vocab,
                POS_vocab=POS_vocab, NER_vocab=NER_vocab)

        self.question_encoder = encoder_utils.SeqEncoder(options, word_vocab,
                POS_vocab=POS_vocab, NER_vocab=NER_vocab, embed_reuse=True)

        with tf.variable_scope('passage'):
            passage_dim, passage_repre, passage_mask = self.passage_encoder.encode(is_training=is_training)
        with tf.variable_scope('question'):
            question_dim, question_repre, question_mask = self.question_encoder.encode(is_training=is_training)

        # modeling entities
        self.entity_starts = tf.placeholder(tf.int32, [None,None], 'entity_starts')
        self.entity_ends = tf.placeholder(tf.int32, [None,None], 'entity_ends')
        self.entity_lengths = tf.placeholder(tf.int32, [None], 'entity_lengths')

        batch_size = tf.shape(self.entity_starts)[0]
        entity_len_max = tf.shape(self.entity_starts)[1]
        entity_mask = tf.sequence_mask(self.entity_lengths, entity_len_max, dtype=tf.float32) # [batch, entity]

        entity_st_rep = operation_utils.collect_node(passage_repre, self.entity_starts) # [batch, entity, rep_dim]
        entity_ed_rep = operation_utils.collect_node(passage_repre, self.entity_ends) # [batch, entity, rep_dim]
        entity_rep = tf.concat([entity_st_rep, entity_ed_rep], axis=2) # [batch, entity, rep_dim * 2]
        entity_dim = passage_dim * 2

        qfull_st_rep = question_repre[:,0,:] # [batch, rep_dim]
        qfull_ed_rep = operation_utils.collect_final_step(question_repre,
                self.question_encoder.sequence_lengths-1) # [batch, rep_dim]
        qfull_rep = tf.concat([qfull_st_rep, qfull_ed_rep], axis=1) # [batch, rep_dim * 2]
        qfull_dim = question_dim * 2

        matching_results = []
        rst_seq = self.perform_matching(entity_rep, entity_dim, entity_mask,
                question_repre, qfull_rep, question_dim, question_mask,
                scope_name='seq_match', options=options, is_training=is_training)
        matching_results.append(rst_seq)

        # encode entity representation with GRN
        if options.with_grn or options.with_gcn:
            # merge question representation into passage
            q4p_rep = tf.tile(tf.expand_dims(qfull_rep, 1), # [batch, 1, rep_dim * 2]
                    [1, entity_len_max, 1]) # [batch, entity, rep_dim * 2]
            entity_rep = tf.concat([entity_rep, q4p_rep], axis=2)
            entity_dim = entity_dim + qfull_dim

            # compress before going to GRN
            merge_w = tf.get_variable('merge_w', [entity_dim, options.merge_dim])
            merge_b = tf.get_variable('merge_b', [options.merge_dim])

            entity_rep = tf.reshape(entity_rep, [-1, entity_dim])
            entity_rep = tf.matmul(entity_rep, merge_w) + merge_b
            entity_rep = tf.reshape(entity_rep, [batch_size, entity_len_max, options.merge_dim])
            entity_rep = entity_rep * tf.expand_dims(entity_mask, axis=-1)
            entity_dim = options.merge_dim

            # main part: encoding
            scope_name = 'GRN' if options.with_grn else 'GCN'

            with tf.variable_scope(scope_name):
                self.edges = tf.placeholder(tf.int32, [None,None,None], 'edges')
                self.edges_mask = tf.placeholder(tf.float32, [None,None,None], 'edges_mask')
                if options.with_grn:
                    print("With Graph recurrent network as the graph encoder")
                    self.graph_encoder = graph_encoder_utils.GraphEncoder(entity_rep, entity_mask, entity_dim,
                            self.edges, self.edges_mask,
                            is_training = is_training, options = options)
                else:
                    print("With GCN as the graph encoder")
                    self.graph_encoder = gcn_encoder_utils.GCNEncoder(entity_rep, entity_mask, entity_dim,
                            self.edges, self.edges_mask,
                            is_training = is_training, options = options)

                for i in range(options.num_grn_step):
                    if options.grn_rep_type == 'hidden':
                        entity_grn_rep = self.graph_encoder.grn_historys[i] # [batch, entity, grn_dim]
                        entity_grn_dim = options.grn_dim
                    elif options.grn_rep_type == 'hidden_embed':
                        entity_grn_rep = tf.concat([self.graph_encoder.grn_historys[i], entity_rep], 2) # [batch, entity, grn_dim + merge_dim]
                        entity_grn_dim = options.grn_dim + entity_dim
                    else:
                        assert False, '%s not supported yet' % options.grn_rep_type

                    if options.with_multi_perspective:
                        assert entity_grn_dim == question_dim

                    rst_grn = self.perform_matching(entity_grn_rep, entity_grn_dim, entity_mask,
                            question_repre, qfull_rep, question_dim, question_mask,
                            scope_name='grn%d_match'%i, options=options, is_training=is_training)
                    matching_results.append(rst_grn)

        self.candidates = tf.placeholder(tf.int32, [None,None,None], 'candidates') # [batch, c_num, c_occur]
        self.candidates_len = tf.placeholder(tf.float32, [None], 'candidates_len') # [batch]
        self.candidates_occur_mask = tf.placeholder(tf.float32, [None,None,None], 'candidates_occur_mask') # [batch, c_num, c_occur]

        # matching_results: list of [batch, cands]
        self.attn_dist = self.perform_integration(matching_results, scope_name='integration',
                options=options, is_training=is_training)

        cand_num = tf.shape(self.candidates)[1]
        self.topk_probs, self.topk_ids = tf.nn.top_k(self.attn_dist, k=cand_num, name='topK')
        self.out = tf.argmax(self.attn_dist, axis=-1, output_type=tf.int32)

        if not has_ref: return

        self.ref = tf.placeholder(tf.int32, [None], 'ref')
        self.accu = tf.reduce_sum(tf.cast(tf.equal(self.out,self.ref), dtype=tf.float32))

        xent = -tf.reduce_sum(
                tf.one_hot(self.ref, cand_num) * tf.log(self.attn_dist),
                axis=-1)

        self.loss = tf.reduce_mean(xent)

        if not is_training: return

        with tf.variable_scope("training_op"), tf.device("/gpu:1"):
            if options.optimize_type == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate=options.learning_rate)
            elif options.optimize_type == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=options.learning_rate)
            clipper = 50 if not options.__dict__.has_key("max_grad_norm") else options.max_grad_norm
            print("Max gradient norm {}".format(clipper))
            tvars = tf.trainable_variables()
            if options.lambda_l2>0.0:
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + options.lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))

            extra_train_ops = []
            train_ops = [self.train_op] + extra_train_ops
            self.train_op = tf.group(*train_ops)


    def perform_matching(self, entity_rep, entity_dim, entity_mask, question_rep, qfull_rep, question_dim, question_mask,
            scope_name, options, is_training):
        with tf.variable_scope(scope_name):
            batch_size = tf.shape(entity_rep)[0]
            entity_num = tf.shape(entity_rep)[1]
            if options.with_multi_perspective:
                assert False, 'Not supported for now'
                #with tf.variable_scope("multi_perspective"):
                #    attn_dist = operation_utils.multi_perspective(entity_rep, self.entity_lengths, entity_mask,
                #            question_repre, self.question_encoder.sequence_lengths, question_mask,
                #            options, is_training)
            else:
                qfull_rep = tf.expand_dims(qfull_rep, 1) # [batch_size, 1, dim]
                qfull_dim = question_dim * 2
                qfull_mask = tf.ones((batch_size, 1)) # [batch_size, 1]
                logits = operation_utils.attention_logits(qfull_rep, entity_rep, qfull_dim, entity_dim, qfull_mask, entity_mask,
                        options=options, is_training=is_training) # [batch, entity]
                logits = tf.reshape(logits, [batch_size, entity_num])
            return logits


    # logits: list of [batch, entity], with list size being 1 or grn_step+1
    def perform_integration(self, matching_results, scope_name, options, is_training):
        with tf.variable_scope(scope_name):
            batch_size = tf.shape(self.candidates)[0]
            if len(matching_results) > 1:
                assert len(matching_results) == options.num_grn_step+1
                matching_results = tf.stack(matching_results, axis=2) # [batch, entity, grn_step+1]
                integrate_w = tf.get_variable('integrate_w', [1, 1, options.num_grn_step+1])
                logits = tf.reduce_sum(matching_results * integrate_w, axis=2) # [batch, entity]
            else:
                assert len(matching_results) == 1
                logits = matching_results[0]

            attn_dist = nn_ops.softmax(logits) # [batch, entity]
            cand_num = tf.shape(self.candidates)[1]
            cand_occur = tf.shape(self.candidates)[2]
            rids = tf.tile(tf.reshape(tf.range(0, limit=batch_size),
                    [-1, 1, 1]),
                        [1, cand_num, cand_occur]) # [batch, c_num, c_occur]
            ids = tf.stack((rids,self.candidates), axis=3) # [batch, c_num, c_occur, 2]
            attn_dist = tf.gather_nd(attn_dist, ids) # [batch, c_num, c_occur]

            attn_dist = attn_dist * self.candidates_occur_mask
            attn_dist = tf.reduce_sum(attn_dist, axis=-1) # [batch, c_num]
            cand_mask = tf.sequence_mask(self.candidates_len, cand_num, dtype=tf.float32) # [batch, c_num]
            attn_dist = attn_dist * cand_mask
            attn_dist = operation_utils.clip_and_normalize(attn_dist, 1.0e-6)
            return attn_dist


    def execute(self, sess, batch, options):
        feed_dict = {}
        feed_dict[self.passage_encoder.sequence_lengths] = batch.passage_len
        feed_dict[self.passage_encoder.sequence_words] = batch.passage
        feed_dict[self.question_encoder.sequence_lengths] = batch.question_len
        feed_dict[self.question_encoder.sequence_words] = batch.question
        if options.with_POS:
            assert False, 'under construction'
        if options.with_NER:
            assert False, 'under construction'
        if options.with_char:
            assert False, 'under construction'
        feed_dict[self.entity_starts] = batch.entity_start
        feed_dict[self.entity_ends] = batch.entity_end
        feed_dict[self.entity_lengths] = batch.entity_len
        if options.with_grn or options.with_gcn:
            feed_dict[self.edges] = batch.entity_edges
            feed_dict[self.edges_mask] = batch.entity_edges_mask
        feed_dict[self.candidates] = batch.cands
        feed_dict[self.candidates_len] = batch.cands_len
        feed_dict[self.candidates_occur_mask] = batch.cands_occur_mask
        if self.has_ref:
            feed_dict[self.ref] = batch.ref

        if self.has_ref == False: # final test
            return sess.run([self.out, self.topk_probs, self.topk_ids], feed_dict)
        else:
            if self.is_training: # train mode
                return sess.run([self.accu, self.loss, self.train_op], feed_dict)
            else: # dev eval
                return sess.run([self.accu, self.loss, self.out], feed_dict)


