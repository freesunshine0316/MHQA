import tensorflow as tf
import os, sys
from elmo.bilm import TokenBatcher, BidirectionalLanguageModel, dump_token_embeddings
import json
import padding_utils
import pickle
import h5py
import time

import collect_vocab_for_MHQA

def process_all_sentences(sentences, sess, batcher, sent_embeddings, sent_token_ids, outpath, batch_size=2, max_length=-1, use_h5=True):
    sentences = sorted(sentences, key=lambda sent: -len(sent[1]))
    batch_spans = padding_utils.make_batches(len(sentences), batch_size)
    all_results = {}
    hf = None
    if use_h5:
        hf = h5py.File(outpath, 'w')

    for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
        cur_ids = []
        cur_sents = []
        cur_lengths = []
        for i in xrange(batch_start, batch_end):
            (cur_id, cur_sent) = sentences[i]
            cur_length = len(cur_sent)
            if max_length != -1:
                if cur_length > max_length:
                    cur_length = max_length
                    cur_sent = cur_sent[:cur_length]
            cur_ids.append(cur_id)
            cur_sents.append(cur_sent)
            cur_lengths.append(cur_length)

        # Create batches of data.
        sent_ids = batcher.batch_sentences(cur_sents)
        # Compute ELMo representations (here for the input only, for simplicity).
        st = time.time()
        elmo_sent_output = sess.run(sent_embeddings, feed_dict={sent_token_ids: sent_ids}) # [batch_size, sent_length, 3*lm_dim]
        print('Length: {}, time: {}'.format(elmo_sent_output.shape, time.time()-st))
        sys.stdout.flush()

        st = time.time()
        for i in xrange(len(cur_ids)):
            cur_id = cur_ids[i]
            cur_length = cur_lengths[i]
            if use_h5:
                if not isinstance(cur_id, unicode):
                    cur_id = cur_id.encode('hex').decode('utf-8')
                if cur_id not in hf.keys():
                    embedding = elmo_sent_output[i,:cur_length,:]
                    hf.create_dataset(cur_id, embedding.shape, dtype='float32', data=embedding)
            else:
                all_results[cur_id] = elmo_sent_output[i,:cur_length,:]
        print('Storing time: {}'.format(time.time()-st))
    if use_h5:
        hf.close()
    else:
        import compress_utils
        compress_utils.save(all_results, outpath)

def generate_ELMo_vector(inpath, out_prefix, elmo_path='/u/zhigwang/zhigwang1/pycham_workspace/MPCM2/src/elmo/tests'):
    print('Loading all questions ...')
    (questions, passages) = collect_vocab_for_MHQA.load_all_instances(inpath)
    print('Number of passages {}, questions {}'.format(len(passages), len(questions)))
    print('DONE!')

    # ============Step 1: create a vocabulary for the SQuAD dataset
    print('Collect all words ...')
    vocab_path = out_prefix + '_vocab.txt'
    collect_vocab_for_MHQA.collect_vocab(questions, passages, vocab_path)
    print('DONE!')

    # ============Step 2: calculate token level embeddings with ELMo
    print('Calculate token level embeddings with ELMo ...')
    # Location of pretrained LM.  Here we use the test fixtures.
    options_file = os.path.join(elmo_path, 'options.json')
    weight_file = os.path.join(elmo_path, 'lm_weights.hdf5')

    # Dump the token embeddings to a file. Run this once for your dataset.
    token_embedding_file = out_prefix + '_elmo_token_embeddings.hdf5'
    dump_token_embeddings(
        vocab_path, options_file, weight_file, token_embedding_file
    )
    tf.reset_default_graph()
    print('DONE!')

    # ============Step 3: calculate ELMo vector for questions and passages
    # Now we can do inference.
    # Create a TokenBatcher to map text to token ids.
    batcher = TokenBatcher(vocab_path)
    sent_token_ids = tf.placeholder('int32', shape=(None, None)) # [batch_size, sent_length]
    # Build the biLM graph.
    bilm = BidirectionalLanguageModel(
        options_file,
        weight_file,
        use_character_inputs=False,
        embedding_weight_file=token_embedding_file
    )

    # Get ops to compute the LM embeddings.
    sent_embeddings = bilm(sent_token_ids)['lm_embeddings'] # [batch_size, 3, sent_length, lm_dim]
    n_lm_layers = int(sent_embeddings.get_shape()[1])
    sent_embeddings = tf.split(sent_embeddings, n_lm_layers, axis=1) # 3 * [batch_size, 1, sent_length, lm_dim]
    sent_embeddings = [tf.squeeze(t, squeeze_dims=1) for t in sent_embeddings]  # 3 * [batch_size, sent_length, lm_dim]
    sent_embeddings = tf.concat(axis=2, values=sent_embeddings) # [batch_size, sent_length, 3*lm_dim]
    with tf.Session() as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        #print('Calculate vectors with ELMo for questions ...')
        #question_elmo_path = out_prefix + '.q.elmovecs.hdf5'
        #process_all_sentences(questions, sess, batcher, sent_embeddings, sent_token_ids, question_elmo_path, batch_size=100)
        #print('DONE!')

        print('Calculate vectors with ELMo for passages ...')
        passage_prefix = out_prefix + '_s_elmovecs'
        for i, cur_passages in enumerate(passages):
            passage_elmo_path = passage_prefix + '/%d.hdf5' % i
            process_all_sentences(cur_passages, sess, batcher, sent_embeddings, sent_token_ids, passage_elmo_path, batch_size=10)
        print('DONE!')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, help='Input file.')
    parser.add_argument('--out_prefix', type=str, help='Prefix for all output files')
    parser.add_argument('--elmo_path', type=str, help='Path to models for EMLo')

    #print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    args, unparsed = parser.parse_known_args()

    generate_ELMo_vector(args.in_path, args.out_prefix, elmo_path=args.elmo_path)


    '''
    inpath = "/u/zhigwang/zhigwang1/tmp/dev.q.elmovecs.pkl"
    # inpath = "/dccstor/arafat1/public/squad/elmo/dev.q.elmovecs.pkl"
    with open(inpath, "r") as f:
        lm_batch = pickle.load(f)

    for key in lm_batch:
        value = lm_batch[key]
        print(key, value.shape)
    #'''




