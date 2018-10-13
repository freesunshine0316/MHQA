import tensorflow as tf
import match_utils

def collect_final_step_lstm(lstm_rep, lens):
    lens = tf.maximum(lens, tf.zeros_like(lens, dtype=tf.int32)) # [batch,]
    idxs = tf.range(0, limit=tf.shape(lens)[0]) # [batch,]
    indices = tf.stack((idxs,lens,), axis=1) # [batch_size, 2]
    return tf.gather_nd(lstm_rep, indices, name='lstm-forward-last')

class SeqEncoder(object):
    def __init__(self, options, word_vocab, char_vocab=None, POS_vocab=None, NER_vocab=None, embed_reuse=None):

        self.options = options

        with tf.variable_scope("embedding_space", reuse=embed_reuse):
            self.word_vocab = word_vocab
            word_vec_trainable = True
            cur_device = '/gpu:0'
            if options.fix_word_vec:
                word_vec_trainable = False
                cur_device = '/cpu:0'
            with tf.variable_scope("embedding"), tf.device(cur_device):
                self.word_embedding = tf.get_variable("word_embedding", trainable=word_vec_trainable,
                        initializer=tf.constant(self.word_vocab.word_vecs), dtype=tf.float32)

            if options.with_char:
                self.char_vocab = char_vocab
                self.char_embedding = tf.get_variable("char_embedding",
                        initializer=tf.constant(self.POS_vocab.word_vecs), dtype=tf.float32)
            else:
                self.char_vocab = None
                self.char_embedding = None

            if options.with_POS:
                self.POS_vocab = POS_vocab
                self.POS_embedding = tf.get_variable("POS_embedding",
                        initializer=tf.constant(self.POS_vocab.word_vecs), dtype=tf.float32)
            else:
                self.POS_vocab = None
                self.POS_embedding = None

            if options.with_NER:
                self.NER_vocab = NER_vocab
                self.NER_embedding = tf.get_variable("NER_embedding",
                        initializer=tf.constant(self.NER_vocab.word_vecs), dtype=tf.float32)
            else:
                self.NER_vocab = None
                self.NER_embedding = None

        self.sequence_lengths = tf.placeholder(tf.int32, [None])
        self.sequence_words = tf.placeholder(tf.int32, [None, None]) # [batch_size, sequence_len]

        if options.with_POS:
            self.sequence_POSs = tf.placeholder(tf.int32, [None, None]) # [batch_size, sequence_len]

        if options.with_NER:
            self.sequence_NERs = tf.placeholder(tf.int32, [None, None]) # [batch_size, sequence_len]

    def encode(self, is_training=True):
        options = self.options

        batch_size = tf.shape(self.sequence_words)[0]
        sequence_len = tf.shape(self.sequence_words)[1]

        # ======word representation layer======
        sequence_repres = []
        input_dim = 0

        word_repres = tf.nn.embedding_lookup(self.word_embedding, self.sequence_words) # [batch_size, sequence_len, word_dim]
        sequence_repres.append(word_repres)
        input_dim += self.word_vocab.word_dim

        if options.with_POS:
            POS_repres = tf.nn.embedding_lookup(self.POS_embedding, self.sequence_POSs) # [batch_size, sequence_len, word_dim]
            sequence_repres.append(POS_repres)
            input_dim += self.POS_vocab.word_dim

        if options.with_NER:
            NER_repres = tf.nn.embedding_lookup(self.NER_embedding, self.sequence_NERs)
            sequence_repres.append(NER_repres)
            input_dim += self.NER_vocab.word_dim

        sequence_repres = tf.concat(sequence_repres, 2) # [batch_size, sequence_len, dim]

        if options.compress_input: # compress input word vector into smaller vectors
            w_compress = tf.get_variable("w_compress_input", [input_dim, options.compress_input_dim], dtype=tf.float32)
            b_compress = tf.get_variable("b_compress_input", [options.compress_input_dim], dtype=tf.float32)

            sequence_repres = tf.reshape(sequence_repres, [-1, input_dim])
            sequence_repres = tf.matmul(sequence_repres, w_compress) + b_compress
            sequence_repres = tf.tanh(sequence_repres)
            sequence_repres = tf.reshape(sequence_repres, [batch_size, sequence_len, options.compress_input_dim])
            input_dim = options.compress_input_dim

        if is_training:
            sequence_repres = tf.nn.dropout(sequence_repres, (1 - options.dropout_rate))

        sequence_mask = tf.sequence_mask(self.sequence_lengths, sequence_len, dtype=tf.float32) # [batch_size, sequence_len]

        # sequential LSTM
        if options.with_cudnn:
            with tf.variable_scope('biCudnnLSTM'):
                sequence_repres = tf.transpose(sequence_repres, [1,0,2]) # [seq_len, batch, dim]
                cudnn_lstm = tf.contrib.cudnn_rnn.CudnnLSTM(options.seq_lstm_layer_num, options.seq_lstm_dim, #input_dim,
                        direction="bidirectional", dropout=options.dropout_rate)
                #input_h = tf.zeros((batch_size, sequence_len, options.seq_lstm_dim))
                #input_c = tf.zeros((batch_size, sequence_len, options.seq_lstm_dim))
                all_sequence_representation, _ = cudnn_lstm(sequence_repres) #, input_h, input_c, None)
                all_sequence_representation = tf.transpose(all_sequence_representation, [1,0,2]) # [batch, seq_len, dim]
                if is_training:
                    all_sequence_representation = tf.nn.dropout(all_sequence_representation, (1 - options.dropout_rate))
            print(tf.shape(all_sequence_representation))
            sequence_dim = options.seq_lstm_dim * 2
        else:
            all_sequence_representation = []
            sequence_dim = 0
            with tf.variable_scope('biLSTM'):
                cur_sequence_repres = sequence_repres
                for i in xrange(options.seq_lstm_layer_num):
                    with tf.variable_scope('layer-{}'.format(i)):
                        # parameters
                        context_lstm_cell_fw = tf.contrib.rnn.LSTMCell(options.seq_lstm_dim)
                        context_lstm_cell_bw = tf.contrib.rnn.LSTMCell(options.seq_lstm_dim)
                        if is_training:
                            context_lstm_cell_fw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_fw, output_keep_prob=(1 - options.dropout_rate))
                            context_lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(context_lstm_cell_bw, output_keep_prob=(1 - options.dropout_rate))

                        # sequence representation
                        ((sequence_context_representation_fw, sequence_context_representation_bw),(_,_)) = tf.nn.bidirectional_dynamic_rnn(
                                    context_lstm_cell_fw, context_lstm_cell_bw, cur_sequence_repres, dtype=tf.float32,
                                    sequence_length=self.sequence_lengths) # [batch_size, sequence_len, seq_lstm_dim]
                        if options.direction == 'forward':
                            # [batch_size, sequence_len, seq_lstm_dim]
                            cur_sequence_repres = sequence_context_representation_fw
                            if i == options.seq_lstm_layer_num-1:
                                sequence_dim += options.seq_lstm_dim
                        elif options.direction == 'backward':
                            # [batch_size, sequence_len, seq_lstm_dim]
                            cur_sequence_repres = sequence_context_representation_bw
                            if i == options.seq_lstm_layer_num-1:
                                sequence_dim += options.seq_lstm_dim
                        elif options.direction == 'bidir':
                            # [batch_size, sequence_len, 2*seq_lstm_dim]
                            cur_sequence_repres = tf.concat(
                                    [sequence_context_representation_fw, sequence_context_representation_bw], 2)
                            if i == options.seq_lstm_layer_num-1:
                                sequence_dim += options.seq_lstm_dim * 2
                        else:
                            assert False
                        if i == options.seq_lstm_layer_num-1:
                            all_sequence_representation.append(cur_sequence_repres)
            all_sequence_representation = tf.concat(all_sequence_representation, 2) # [batch_size, sequence_len, sequence_dim]

        all_sequence_representation = all_sequence_representation * tf.expand_dims(sequence_mask, axis=-1)

        return (sequence_dim, all_sequence_representation, sequence_mask)

