# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import re
import os
import sys
import json
import time
import numpy as np
import codecs

import tensorflow as tf
import namespace_utils

import MHQA_trainer
import MHQA_data_stream
from MHQA_model_graph import ModelGraph
from vocab_utils import Vocab


tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_prefix', type=str, required=True, help='Prefix to the models.')
    parser.add_argument('--in_path', type=str, required=True, help='The path to the test file.')
    parser.add_argument('--out_path', type=str, help='The path to the output file.')

    args, unparsed = parser.parse_known_args()

    model_prefix = args.model_prefix
    in_path = args.in_path
    out_path = args.out_path

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    # load the configuration file
    print('Loading configurations from ' + model_prefix + ".config.json")
    FLAGS = namespace_utils.load_namespace(model_prefix + ".config.json")
    FLAGS = MHQA_trainer.enrich_options(FLAGS)

    # load vocabs
    print('Loading vocabs.')
    word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
    print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
    char_vocab = None
    if FLAGS.with_char:
        char_vocab = Vocab(model_prefix + ".char_vocab", fileformat='txt2')
        print('char_vocab: {}'.format(char_vocab.word_vecs.shape))


    print('Loading test set from {}.'.format(in_path))
    testset, _ = MHQA_data_stream.read_data_file(in_path, FLAGS, force_noref=True)
    print('Number of samples: {}'.format(len(testset)))

    print('Build DataStream ... ')
    testDataStream = MHQA_data_stream.DataStream(testset, word_vocab, char_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=True, has_ref=False)
    print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
    print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))

    best_path = model_prefix + ".best.model"
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-0.01, 0.01)
        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=False, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, char_vocab=char_vocab, options=FLAGS,
                        has_ref=False, is_training=False)

        ## remove word _embedding
        vars_ = {}
        for var in tf.all_variables():
            if FLAGS.fix_word_vec and "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
        saver = tf.train.Saver(vars_)

        initializer = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(initializer)

        saver.restore(sess, best_path) # restore the model

        test_total = 0
        test_right = 0
        outfile = codecs.open(out_path, 'wt', 'utf-8')
        total_num = testDataStream.get_num_batch()
        testDataStream.reset()
        for i in range(total_num):
            cur_batch = testDataStream.get_batch(i)
            cur_out, cur_topk_probs, cur_topk_ids = valid_graph.execute(sess, cur_batch, FLAGS)
            # output k-best
            for j in range(cur_batch.batch_size):
                cur_topk_cands = [cur_batch.candidates_str[j][x] \
                        for x in cur_topk_ids[j] if x < len(cur_batch.candidates_str[j])]
                cur_qid = cur_batch.ids[j]
                cur_json_str = json.dumps({'question_id': cur_qid, 'ranked_cands': cur_topk_cands})
                outfile.write(cur_json_str+'\n')
            # compare 1-best
            cur_out = [cur_batch.candidates_str[j][idx] for j, idx in enumerate(cur_out)]
            if cur_batch.ref_str != None:
                test_right += sum(out == cur_batch.ref_str[j] for j, out in enumerate(cur_out))
                test_total += cur_batch.batch_size
        if test_total != 0:
            print('Evaluation accuracy {}'.format(1.0*test_right/test_total))
        outfile.close()

