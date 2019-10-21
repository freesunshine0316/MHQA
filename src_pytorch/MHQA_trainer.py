# -*- coding: utf-8 -*-
import argparse
import os
import sys
import time
import numpy as np
import codecs
import random

import namespace_utils
import MHQA_data_stream
import MHQA_model_graph

FLAGS = None

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.optimization import BertAdam
from torch.optim import Adam

def evaluate_dataset(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = []
    ref = []
    dev_loss = 0.0
    dev_right = 0.0
    dev_total = 0.0
    for step, ori_batch in enumerate(dataset):
        batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                for k, v in ori_batch.items()}
        outputs = model(batch)
        dev_loss += outputs['loss'].item()
        dev_right += outputs['right_count'].item()
        dev_total += len(batch['ids'])

    return {'dev_loss':dev_loss, 'dev_accu':1.0*dev_right/dev_total, 'dev_right':dev_right, 'dev_total':dev_total, }



def main():
    print(FLAGS.__dict__)
    log_dir = FLAGS.log_dir
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}, grad_accum_steps: {}'.format(device, n_gpu, FLAGS.grad_accum_steps))
    log_file.write('device: {}, n_gpu: {}, grad_accum_steps: {}\n'.format(device, n_gpu, FLAGS.grad_accum_steps))

    print('Loading train set.')
    trainset, _ = MHQA_data_stream.read_data_file(FLAGS.train_path, FLAGS)
    trainset_batches = MHQA_data_stream.make_batches_elmo(trainset, FLAGS)
    print('Number of training samples: {}'.format(len(trainset)))
    print('Number of training batches: {}'.format(len(trainset_batches)))

    print('Loading dev set.')
    devset, _ = MHQA_data_stream.read_data_file(FLAGS.dev_path, FLAGS)
    devset_batches = MHQA_data_stream.make_batches_elmo(devset, FLAGS)
    print('Number of dev samples: {}'.format(len(devset)))
    print('Number of dev batches: {}'.format(len(devset_batches)))

    # model
    print('Compiling model.')
    model = MHQA_model_graph.ModelGraph(FLAGS)
    if os.path.exists(path_prefix + ".model.bin"):
        print('!!Existing pretrained model. Loading the model...')
        model.load_state_dict(torch.load(path_prefix + ".model.bin"))
    model.to(device)

    # pretrained performance
    best_accu = 0.0
    if os.path.exists(path_prefix + ".model.bin"):
        best_accu = FLAGS.best_accu if 'best_accu' in FLAGS.__dict__ and abs(FLAGS.best_accu) > 1e-4 \
                else evaluate_dataset(model, devset_batches)
        FLAGS.best_accu = best_accu
        print("!!Accuracy for pretrained model is {}".format(best_accu))

    # optimizer
    train_updates = len(trainset_batches) * FLAGS.num_epochs
    if FLAGS.grad_accum_steps > 1:
        train_updates = train_updates // FLAGS.grad_accum_steps
    if FLAGS.optim == 'bertadam':
        optimizer = BertAdam(model.parameters(),
                lr=FLAGS.learning_rate, warmup=FLAGS.warmup_proportion, t_total=train_updates)
    elif FLAGS.optim == 'adam':
        optimizer = Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.lambda_l2)
    else:
        assert False, 'unsupported optimizer type: {}'.format(FLAGS.optim)

    print('Start the training loop, total *updating* steps = {}'.format(train_updates))
    finished_steps, finished_epochs = 0, 0
    train_batch_ids = list(range(0, len(trainset_batches)))
    model.train()
    while finished_epochs < FLAGS.num_epochs:
        epoch_start = time.time()
        epoch_loss = []
        print('Current epoch takes {} steps'.format(len(train_batch_ids)))
        random.shuffle(train_batch_ids)
        start_time = time.time()
        for id in train_batch_ids:
            ori_batch = trainset_batches[id]
            batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                    for k, v in ori_batch.items()}

            outputs = model(batch)
            loss = outputs['loss']
            epoch_loss.append(loss.item())

            if n_gpu > 1:
                loss = loss.mean()
            if FLAGS.grad_accum_steps > 1:
                loss = loss / FLAGS.grad_accum_steps
            loss.backward() # just calculate gradient

            finished_steps += 1
            if finished_steps % FLAGS.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if finished_steps % 100==0:
                print('{} '.format(finished_steps), end="")
                sys.stdout.flush()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save a checkpoint and evaluate the model periodically.
            if finished_steps > 0 and finished_steps % 1000 == 0:
                best_accu = validate_and_save(model, devset_batches, log_file, best_accu)
        duration = time.time() - start_time
        print('Training loss = %.2f (%.3f sec)' % (float(sum(epoch_loss)), duration))
        log_file.write('Training loss = %.2f (%.3f sec)\n' % (float(sum(epoch_loss)), duration))
        finished_epochs += 1
        best_accu = validate_and_save(model, devset_batches, log_file, best_accu)

    log_file.close()


def validate_and_save(model, devset_batches, log_file, best_accu):
    path_prefix = FLAGS.log_dir + "/MHQA.{}".format(FLAGS.suffix)
    start_time = time.time()
    res_dict = evaluate_dataset(model, devset_batches)
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
        best_path = path_prefix + '.model.bin'
        torch.save(model.state_dict(), best_path)
        best_accu = dev_accu
        FLAGS.best_accu = dev_accu
        namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
    duration = time.time() - start_time
    print('Duration %.3f sec' % (duration))
    print('-------------')
    log_file.write('-------------\n')
    return best_accu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.config_path is not None:
        print('Loading the configuration from ' + FLAGS.config_path)
        FLAGS = namespace_utils.load_namespace(FLAGS.config_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])

    sys.stdout.flush()
    main()
