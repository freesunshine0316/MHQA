# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import numpy as np
import codecs

from vocab_utils import Vocab
import namespace_utils
import MHQA_data_stream
from MHQA_model_graph import ModelGraph

FLAGS = None
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # DEBUG, INFO, WARN, ERROR, and FATAL


import platform
def get_machine_name():
    return platform.node()

def vec2string(val):
    result = ""
    for v in val:
        result += " {}".format(v)
    return result.strip()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def evaluate(sess, valid_graph, devDataStream, options=None, suffix=''):
    devDataStream.reset()
    gen = []
    ref = []
    dev_loss = 0.0
    dev_right = 0.0
    dev_total = 0.0
    for batch_index in xrange(devDataStream.get_num_batch()): # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        cur_batch = MHQA_data_stream.BatchPadded(cur_batch)
        cur_accu, cur_loss, _ = valid_graph.execute(sess, cur_batch, options)
        dev_loss += cur_loss
        dev_right += cur_accu
        dev_total += cur_batch.batch_size

    return {'dev_loss':dev_loss, 'dev_accu':1.0*dev_right/dev_total, 'dev_right':dev_right, 'dev_total':dev_total, }



def main(_):
    print('Configurations:')
    print(FLAGS)

    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/MHQA.{}".format(FLAGS.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(FLAGS))
    log_file.flush()

    # save configuration
    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")

    print('Loading train set.')
    trainset, _, _ = MHQA_data_stream.read_data_file(FLAGS.train_path, FLAGS)
    print('Number of training samples: {}'.format(len(trainset)))

    print('Loading dev set.')
    devset, _, _ = MHQA_data_stream.read_data_file(FLAGS.dev_path, FLAGS)
    print('Number of dev samples: {}'.format(len(devset)))

    word_vocab = None
    char_vocab = None
    has_pretrained_model = False
    best_path = path_prefix + ".best.model"
    if os.path.exists(best_path + ".index"):
        has_pretrained_model = True
        print('!!Existing pretrained model. Loading vocabs.')
        word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
        print('word_vocab: {}'.format(word_vocab.word_vecs.shape))
        char_vocab = None
        if FLAGS.with_char:
            char_vocab = Vocab(path_prefix + ".char_vocab", fileformat='txt2')
            print('char_vocab: {}'.format(char_vocab.word_vecs.shape))
    else:
        print('Collecting vocabs.')
        (allWords, allChars) = MHQA_data_stream.collect_vocabs(trainset)
        print('Number of words: {}'.format(len(allWords)))
        print('Number of allChars: {}'.format(len(allChars)))

        word_vocab = Vocab(FLAGS.word_vec_path, fileformat='txt2')
        char_vocab = None
        if FLAGS.with_char:
            char_vocab = Vocab(voc=allChars, dim=FLAGS.char_dim, fileformat='build')
            char_vocab.dump_to_txt2(path_prefix + ".char_vocab")

    print('word vocab size {}'.format(word_vocab.vocab_size))
    sys.stdout.flush()

    print('Build DataStream ... ')
    trainDataStream = MHQA_data_stream.DataStream(trainset, word_vocab, char_vocab, options=FLAGS,
                 isShuffle=True, isLoop=True, isSort=True, has_ref=True)

    devDataStream = MHQA_data_stream.DataStream(devset, word_vocab, char_vocab, options=FLAGS,
                 isShuffle=False, isLoop=False, isSort=True, has_ref=True)
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))

    sys.stdout.flush()

    # initialize the best bleu and accu scores for current training session
    best_accu = FLAGS.best_accu if FLAGS.__dict__.has_key('best_accu') else 0.0
    if best_accu > 0.0:
        print('With initial dev accuracy {}'.format(best_accu))

    init_scale = 0.01
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        with tf.name_scope("Train"):
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_graph = ModelGraph(word_vocab=word_vocab, char_vocab=char_vocab, options=FLAGS,
                        has_ref=True, is_training=True)

        with tf.name_scope("Valid"):
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_graph = ModelGraph(word_vocab=word_vocab, char_vocab=char_vocab, options=FLAGS,
                        has_ref=True, is_training=False)

        initializer = tf.global_variables_initializer()

        vars_ = {}
        for var in tf.all_variables():
            if FLAGS.fix_word_vec and "word_embedding" in var.name: continue
            if not var.name.startswith("Model"): continue
            vars_[var.name.split(":")[0]] = var
            print(var)
        saver = tf.train.Saver(vars_)

        sess = tf.Session()
        sess.run(initializer)
        if has_pretrained_model:
            print("Restoring model from " + best_path)
            saver.restore(sess, best_path)
            print("DONE!")

            if abs(best_accu) < 0.00001:
                print("Getting ACCU score for the model")
                best_accu = evaluate(sess, valid_graph, devDataStream, options=FLAGS)['dev_accu']
                FLAGS.best_accu = best_accu
                namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
                print('ACCU = %.4f' % best_accu)
                log_file.write('ACCU = %.4f\n' % best_accu)

        print('Start the training loop.')
        train_size = trainDataStream.get_num_batch()
        max_steps = train_size * FLAGS.max_epochs
        total_loss = 0.0
        start_time = time.time()
        for step in xrange(max_steps):
            cur_batch = trainDataStream.nextBatch()
            cur_batch = MHQA_data_stream.BatchPadded(cur_batch)
            _, cur_loss, _ = train_graph.execute(sess, cur_batch, FLAGS)
            total_loss += cur_loss

            if step % 100==0:
                print('{} '.format(step), end="")
                sys.stdout.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % trainDataStream.get_num_batch() == 0 or (step + 1) == max_steps or \
                    (trainDataStream.get_num_batch() > 10000 and (step+1)%2000 == 0):
                print()
                duration = time.time() - start_time
                print('Step %d: loss = %.2f (%.3f sec)' % (step, total_loss, duration))
                log_file.write('Step %d: loss = %.2f (%.3f sec)\n' % (step, total_loss, duration))
                log_file.flush()
                sys.stdout.flush()
                total_loss = 0.0

                best_accu = validate_and_save(sess, saver, FLAGS, log_file,
                        devDataStream, valid_graph, path_prefix, best_accu)
                start_time = time.time()

    log_file.close()


def validate_and_save(sess, saver, FLAGS, log_file,
        devDataStream, valid_graph, path_prefix, best_accu):
    best_path = path_prefix + ".best.model"
    start_time = time.time()
    print('Validation Data Eval:')
    res_dict = evaluate(sess, valid_graph, devDataStream, options=FLAGS)
    dev_loss = res_dict['dev_loss']
    dev_accu = res_dict['dev_accu']
    dev_right = int(res_dict['dev_right'])
    dev_total = int(res_dict['dev_total'])
    print('Dev loss = %.4f' % dev_loss)
    log_file.write('Dev loss = %.4f\n' % dev_loss)
    print('Dev accu = %.4f %d/%d' % (dev_accu, dev_right, dev_total))
    log_file.write('Dev accu = %.4f %d/%d\n' % (dev_accu, dev_right, dev_total))
    log_file.flush()
    if best_accu < dev_accu:
        print('Saving weights, ACCU {} (prev_best) < {} (cur)'.format(best_accu, dev_accu))
        saver.save(sess, best_path)
        best_accu = dev_accu
        FLAGS.best_accu = dev_accu
        namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
    duration = time.time() - start_time
    print('Duration %.3f sec' % (duration))
    sys.stdout.flush()
    return best_accu


def enrich_options(options):
    if not options.__dict__.has_key("with_same"):
        options.with_same = True

    if not options.__dict__.has_key("with_coref"):
        options.with_coref = True

    if not options.__dict__.has_key("with_window"):
        options.with_window = True

    return options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')

    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    #os.environ["CUDA_VISIBLE_DEVICES"]="2"

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    FLAGS, unparsed = parser.parse_known_args()


    if FLAGS.config_path is not None:
        print('Loading the configuration from ' + FLAGS.config_path)
        FLAGS = namespace_utils.load_namespace(FLAGS.config_path)

    FLAGS = enrich_options(FLAGS)

    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
