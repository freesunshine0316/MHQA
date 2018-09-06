import tensorflow as tf
import layer_utils

def multi_perspective_match(feature_dim, repres1, repres2, is_training=True, dropout_rate=0.2,
                            options=None, scope_name='mp-match', reuse=False):
    '''
        :param repres1: [batch_size, len, feature_dim]
        :param repres2: [batch_size, len, feature_dim]
        :return:
    '''
    repres1 = layer_utils.dropout_layer(repres1, dropout_rate, is_training=is_training)
    repres2 = layer_utils.dropout_layer(repres2, dropout_rate, is_training=is_training)
    input_shape = tf.shape(repres1)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    matching_result = []
    cosine_norm = True
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0
        if options.with_cosine:
            cosine_value = layer_utils.cosine_distance(repres1, repres2, cosine_norm=cosine_norm)
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            matching_result.append(cosine_value)
            match_dim += 1

        concat_rep = tf.concat(axis=2, values=[repres1, repres2])
        if options.with_nn_match:
            nn_match_W = tf.get_variable("nn_match_W", [2 * feature_dim, options.nn_match_dim], dtype=tf.float32)
            nn_match_b = tf.get_variable("nn_match_b", [options.nn_match_dim], dtype=tf.float32)
            cur_rep = tf.reshape(concat_rep, [batch_size * seq_length, 2 * feature_dim])
            cur_match_result = tf.tanh(tf.matmul(cur_rep, nn_match_W) + nn_match_b)
            cur_match_result = tf.reshape(cur_match_result, [batch_size, seq_length, options.nn_match_dim])
            matching_result.append(cur_match_result)
            match_dim += options.nn_match_dim

        if options.with_mp_cosine:
            if options.mp_cosine_proj_dim > 0:
                mp_cosine_projection = tf.get_variable("mp_cosine_projection",
                                    [feature_dim, options.mp_cosine_proj_dim], dtype=tf.float32)
                mp_cosine_params = tf.get_variable("mp_cosine", shape=[1, options.cosine_MP_dim,
                                                 options.mp_cosine_proj_dim], dtype=tf.float32)
                repres1_flat = tf.reshape(repres1, [batch_size * seq_length, feature_dim])
                repres2_flat = tf.reshape(repres2, [batch_size * seq_length, feature_dim])
                repres1_flat = tf.tanh(tf.matmul(repres1_flat, mp_cosine_projection))
                repres2_flat = tf.tanh(tf.matmul(repres2_flat, mp_cosine_projection))
                repres1_flat = tf.expand_dims(repres1_flat, axis=1)
                repres2_flat = tf.expand_dims(repres2_flat, axis=1)
                mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(repres1_flat, mp_cosine_params), repres2_flat,
                                                                 cosine_norm=cosine_norm)
                mp_cosine_matching = tf.reshape(mp_cosine_matching, [batch_size, seq_length, options.cosine_MP_dim])
            else:
                mp_cosine_params = tf.get_variable("mp_cosine", shape=[1, 1, options.cosine_MP_dim, feature_dim], dtype=tf.float32)
                repres1_flat = tf.expand_dims(repres1, axis=2)
                repres2_flat = tf.expand_dims(repres2, axis=2)
                mp_cosine_matching = layer_utils.cosine_distance(tf.multiply(repres1_flat, mp_cosine_params),
                                                                 repres2_flat, cosine_norm=cosine_norm)
            matching_result.append(mp_cosine_matching)
            match_dim += options.cosine_MP_dim

        if options.with_match_lstm:
            (_, _, match_lstm_result) = layer_utils.my_lstm_layer(concat_rep, options.match_lstm_dim,
                                scope_name="match_lstm", reuse=False, is_training=is_training, dropout_rate=dropout_rate)
            matching_result.append(match_lstm_result)
            match_dim += 2 * options.match_lstm_dim
    matching_result = tf.concat(axis=2, values=matching_result)
    return (matching_result, match_dim)

def multi_granularity_match(feature_dim, passage, question, passage_length, question_length,
                            passage_mask=None, question_mask=None,is_training=True, dropout_rate=0.2,
                            options=None, with_full_matching=False, with_attentive_matching=True,
                            with_max_attentive_matching=True, scope_name='mgm', reuse=False):
    '''
        passage: [batch_size, passage_length, feature_dim]
        question: [batch_size, question_length, feature_dim]
        passage_length: [batch_size]
        question_length: [batch_size]
    '''
    input_shape = tf.shape(passage)
    batch_size = input_shape[0]
    passage_len = input_shape[1]

    match_reps = []
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0
        if with_full_matching:
            passage_fw = passage[:, :, 0:feature_dim / 2]
            passage_bw = passage[:, :, feature_dim / 2:feature_dim]

            question_fw = question[:, :, 0:feature_dim / 2]
            question_bw = question[:, :, feature_dim / 2:feature_dim]
            question_fw = layer_utils.collect_final_step_of_lstm(question_fw, question_length - 1)  # [batch_size, feature_dim/2]
            question_bw = question_bw[:, 0, :]

            question_fw = tf.expand_dims(question_fw, axis=1)
            question_fw = tf.tile(question_fw, [1, passage_len, 1])  # [batch_size, pasasge_len, feature_dim]

            question_bw = tf.expand_dims(question_bw, axis=1)
            question_bw = tf.tile(question_bw, [1, passage_len, 1])  # [batch_size, pasasge_len, feature_dim]
            (fw_full_match_reps, fw_full_match_dim) = multi_perspective_match(feature_dim / 2, passage_fw, question_fw,
                                                    is_training=is_training, dropout_rate=dropout_rate, options=options,
                                                                              scope_name='fw_full_match')
            (bw_full_match_reps, bw_full_match_dim) = multi_perspective_match(feature_dim / 2, passage_bw, question_bw,
                                                    is_training=is_training, dropout_rate=dropout_rate, options=options,
                                                                              scope_name='bw_full_match')
            match_reps.append(fw_full_match_reps)
            match_reps.append(bw_full_match_reps)
            match_dim += fw_full_match_dim
            match_dim += bw_full_match_dim

        if with_attentive_matching or with_max_attentive_matching:
            atten_scores = layer_utils.calcuate_attention(passage, question, feature_dim, feature_dim, scope_name="attention",
                       att_type=options.attn_type, att_dim=options.attn_depth, remove_diagnoal=False,
                                                          mask1=passage_mask, mask2=question_mask,
                                                          is_training=is_training, dropout_rate=dropout_rate)
            # match_reps.append(tf.reduce_max(atten_scores, axis=2, keep_dims=True))
            # match_reps.append(tf.reduce_mean(atten_scores, axis=2, keep_dims=True))
            # match_dim += 2

        if with_max_attentive_matching:
            atten_positions = tf.argmax(atten_scores, axis=2, output_type=tf.int32)  # [batch_size, passage_len]
            max_question_reps = layer_utils.collect_representation(question, atten_positions)
            (max_att_match_rep, max_att_match_dim) = multi_perspective_match(feature_dim, passage, max_question_reps,
                                        is_training=is_training, dropout_rate=dropout_rate, options=options, scope_name='max_att_match')
            match_reps.append(max_att_match_rep)
            match_dim += max_att_match_dim

        if with_attentive_matching:
            att_rep = tf.matmul(atten_scores, question)
            (attentive_match_rep, attentive_match_dim) = multi_perspective_match(feature_dim, passage, att_rep,
                                                    is_training=is_training, dropout_rate=dropout_rate, options=options,
                                                                                 scope_name='att_match')
            match_reps.append(attentive_match_rep)
            match_dim += attentive_match_dim
    match_reps = tf.concat(axis=2, values=match_reps)
    return (match_reps, match_dim)


def MPCM_match(feature_dim, passage, question, passage_length, question_length, passage_mask, question_mask,
               options=None, scope_name='MPCM_match_layer', is_training=True, dropout_rate=0.2, reuse=False):
    match_results = []
    match_dim = 0
    with tf.variable_scope(scope_name, reuse=reuse):
        if options.with_word_match:
            (word_match_reps, word_match_dim) = multi_granularity_match(feature_dim, passage, question, passage_length, question_length,
                            passage_mask=passage_mask, question_mask=question_mask,is_training=is_training, dropout_rate=dropout_rate,
                            options=options, with_full_matching=False, with_attentive_matching=True,
                            with_max_attentive_matching=True, scope_name='word_match', reuse=False)
            match_results.append(word_match_reps)
            match_dim += word_match_dim

        if options.with_sequential_match:
            cur_passage_context = None
            cur_question_context = None
            for i in xrange(options.context_layer_num):
                if cur_passage_context is None:
                    cur_passage_context = passage
                    cur_question_context = question
                else:
                    cur_passage_context = tf.concat(axis=2, values=[passage, cur_passage_context])
                    cur_question_context = tf.concat(axis=2, values=[question, cur_question_context])
                (cur_passage_context_fw, cur_passage_context_bw, cur_passage_context) = layer_utils.my_lstm_layer(
                        cur_passage_context, options.context_lstm_dim, scope_name="passage_context_lstm_{}".format(i),
                        reuse=False, is_training=is_training, dropout_rate=dropout_rate)
                cur_passage_context_fw = tf.multiply(cur_passage_context_fw, tf.expand_dims(passage_mask, axis=-1))
                cur_passage_context_bw = tf.multiply(cur_passage_context_bw, tf.expand_dims(passage_mask, axis=-1))
                cur_passage_context = tf.multiply(cur_passage_context, tf.expand_dims(passage_mask, axis=-1))

                (cur_question_context_fw, cur_question_context_bw, cur_question_context) = layer_utils.my_lstm_layer(
                    cur_question_context, options.context_lstm_dim, scope_name="question_context_lstm_{}".format(i),
                    reuse=False, is_training=is_training, dropout_rate=dropout_rate)
                cur_question_context_fw = tf.multiply(cur_question_context_fw, tf.expand_dims(question_mask, axis=-1))
                cur_question_context_bw = tf.multiply(cur_question_context_bw, tf.expand_dims(question_mask, axis=-1))
                cur_question_context = tf.multiply(cur_question_context, tf.expand_dims(question_mask, axis=-1))

                if options.with_attentive_match:
                    # forward matching
                    (cur_match_rep, cur_match_dim) = multi_granularity_match(options.context_lstm_dim,
                            cur_passage_context_fw, cur_question_context_fw, passage_length, question_length,
                            passage_mask=passage_mask, question_mask=question_mask,is_training=is_training,
                            dropout_rate=dropout_rate, options=options, with_full_matching=False,
                            with_attentive_matching=True, with_max_attentive_matching=True, scope_name='seq_forward_match_{}'.format(i))
                    match_dim += cur_match_dim
                    match_results.append(cur_match_rep)

                    # backward matching
                    (cur_match_rep, cur_match_dim) = multi_granularity_match(options.context_lstm_dim,
                            cur_passage_context_bw, cur_question_context_bw, passage_length, question_length,
                            passage_mask=passage_mask, question_mask=question_mask,is_training=is_training,
                            dropout_rate=dropout_rate, options=options, with_full_matching=False, with_attentive_matching=True,
                            with_max_attentive_matching=True, scope_name='seq_backward_match_{}'.format(i))
                    match_dim += cur_match_dim
                    match_results.append(cur_match_rep)

                if options.with_full_match:
                    # full matching
                    (cur_match_rep, cur_match_dim) = multi_granularity_match(2*options.context_lstm_dim,
                            cur_passage_context, cur_question_context,
                            passage_length, question_length, passage_mask=passage_mask, question_mask=question_mask,is_training=is_training,
                            dropout_rate=dropout_rate, options=options, with_full_matching=True, with_attentive_matching=False,
                            with_max_attentive_matching=False, scope_name='seq_full_match_{}'.format(i))
                    match_dim += cur_match_dim
                    match_results.append(cur_match_rep)

                if options.with_word_phrase_match:
                    question_context_proj = layer_utils.projection_layer(cur_question_context, 2*options.context_lstm_dim, feature_dim,
                                                                 activation_func=tf.tanh, scope="question_context_proj_{}".format(i))
                    (cur_match_rep, cur_match_dim) = multi_granularity_match(feature_dim, passage, question_context_proj,
                                passage_length, question_length, passage_mask=passage_mask, question_mask=question_mask,
                                is_training=is_training, dropout_rate=dropout_rate, options=options, with_full_matching=False,
                                with_attentive_matching=True, with_max_attentive_matching=True, scope_name='word_phrase_match_{}'.format(i))

                    match_dim += cur_match_dim
                    match_results.append(cur_match_rep)


                if options.with_phrase_word_match:
                    passage_context_proj = layer_utils.projection_layer(cur_passage_context, 2*options.context_lstm_dim, feature_dim,
                                                                 activation_func=tf.tanh, scope="passage_context_proj_{}".format(i))
                    (cur_match_rep, cur_match_dim) = multi_granularity_match(feature_dim, passage_context_proj, question,
                            passage_length, question_length, passage_mask=passage_mask, question_mask=question_mask,is_training=is_training,
                            dropout_rate=dropout_rate, options=options, with_full_matching=False, with_attentive_matching=True,
                            with_max_attentive_matching=True, scope_name='phrase_word_match_{}'.format(i))
                    match_dim += cur_match_dim
                    match_results.append(cur_match_rep)

    match_results = tf.concat(axis=2, values=match_results)
    return (match_results, match_dim)


def FusionNet_match(feature_dim, passage, question, passage_length, question_length, passage_mask, question_mask, onehot_binary=None,
               options=None, scope_name='FusionNet_match_layer', is_training=True, dropout_rate=0.2, reuse=False):
    # passage_mask = None
    # question_mask = None

    with tf.variable_scope(scope_name, reuse=reuse):
        #======= Fully Aware MultiLevel Fusion (FAMF) Word  Layer
        # word_atten_scores = layer_utils.calcuate_attention \
        word_atten_scores = layer_utils.calcuate_attention(passage, question, feature_dim, feature_dim, scope_name="FAMF_word",
                                                  att_type=options.att_type, att_dim=options.att_dim, remove_diagnoal=False,
                                                           mask1=passage_mask, mask2=question_mask,
                                                           is_training=is_training, dropout_rate=dropout_rate)
        weighted_by_question_words = tf.matmul(word_atten_scores, layer_utils.dropout_layer(question, dropout_rate, is_training=is_training))

        #====== Reading layer
        passage_tmp = [passage, weighted_by_question_words]
        passage_tmp_dim = 2 * feature_dim
        if onehot_binary is not None:
            passage_tmp.append(onehot_binary)
            passage_tmp_dim += 11
        passage_tmp = tf.concat(axis=2, values=passage_tmp)
        passage_context1 = layer_utils.my_lstm_layer(passage_tmp, options.context_lstm_dim,
                                    scope_name="passage_context1_lstm", reuse=False, is_training=is_training, dropout_rate=dropout_rate)[2]
        passage_context2 = layer_utils.my_lstm_layer(passage_context1, options.context_lstm_dim,
                                    scope_name="passage_context2_lstm", reuse=False, is_training=is_training, dropout_rate=dropout_rate)[2]

        question_context1 = layer_utils.my_lstm_layer(question, options.context_lstm_dim,
                                    scope_name="question_context1_lstm", reuse=False, is_training=is_training, dropout_rate=dropout_rate)[2]
        question_context2 = layer_utils.my_lstm_layer(question_context1, options.context_lstm_dim,
                                    scope_name="question_context2_lstm", reuse=False, is_training=is_training, dropout_rate=dropout_rate)[2]

        # ==== Understanding Layer
        quesiton_understand_input = tf.concat(axis=2, values=(question_context1, question_context2))
        quesiton_understand_output = layer_utils.my_lstm_layer(quesiton_understand_input, options.context_lstm_dim,
                                    scope_name="question_under_lstm", reuse=False, is_training=is_training, dropout_rate=dropout_rate)[2]

        # ==== FAMF : higher level
        famf_passage_input = tf.concat(axis=2, values=(passage, passage_context1, passage_context2))
        famf_question_input = tf.concat(axis=2, values=(question, question_context1, question_context2))

        passage_in_dim = feature_dim + 4 * options.context_lstm_dim

        lower_level_atten_scores = layer_utils.calcuate_attention(famf_passage_input, famf_question_input, passage_in_dim, passage_in_dim,
                                                                  scope_name="lower_level_att",
                                        att_type=options.att_type, att_dim=options.att_dim, remove_diagnoal=False,
                                        mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)
        high_level_atten_scores = layer_utils.calcuate_attention(famf_passage_input, famf_question_input, passage_in_dim, passage_in_dim,
                                                                 scope_name="high_level_att",
                                        att_type=options.att_type, att_dim=options.att_dim, remove_diagnoal=False,
                                        mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)
        understand_atten_scores = layer_utils.calcuate_attention(famf_passage_input, famf_question_input, passage_in_dim, passage_in_dim,
                                                                 scope_name="understand_att",
                                        att_type=options.att_type, att_dim=options.att_dim, remove_diagnoal=False,
                                        mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)
        h_Cl = tf.matmul(lower_level_atten_scores, layer_utils.dropout_layer(question_context1, dropout_rate, is_training=is_training))
        h_Ch = tf.matmul(high_level_atten_scores, layer_utils.dropout_layer(question_context2, dropout_rate, is_training=is_training))
        u_C = tf.matmul(understand_atten_scores, layer_utils.dropout_layer(quesiton_understand_output, dropout_rate, is_training=is_training))

        # ====famf_higher_layer_passage_lstm
        V_c_input = tf.concat(axis=2, values=[passage_context1, passage_context2, h_Cl, h_Ch, u_C])
        V_c = layer_utils.my_lstm_layer(V_c_input, options.context_lstm_dim, scope_name="famf_higher_layer_passage_lstm",
                                        reuse=False, is_training=is_training, dropout_rate=dropout_rate)[2]

        # VV_c_input = tf.concat(axis=2, values=[passage_tmp, V_c_input, V_c])
        # input_dim = 12*options.context_lstm_dim + passage_tmp_dim
        VV_c_input = tf.concat(axis=2, values=[passage, V_c_input, V_c])
        input_dim = 12*options.context_lstm_dim + feature_dim
        # ==== FAMF: Self-boosted
        if options.with_self_match:
            VV_c_input_projection = layer_utils.projection_layer(VV_c_input, input_dim, options.self_compress_dim,
                                                             scope="self-boost-projection")
            self_atten_scores = layer_utils.calcuate_attention(VV_c_input_projection, VV_c_input_projection, options.self_compress_dim,                                         options.self_compress_dim,
                                    scope_name="self_boost_att", att_type=options.att_type, att_dim=options.att_dim,
                                                               remove_diagnoal=options.remove_diagonal,
                                    mask1=passage_mask, mask2=passage_mask, is_training=is_training, dropout_rate=dropout_rate)
            VV_c = tf.matmul(self_atten_scores, layer_utils.dropout_layer(V_c, dropout_rate, is_training=is_training))
            VV_c_input = tf.concat(axis=2, values=[VV_c_input, VV_c])
            input_dim += 2*options.context_lstm_dim
        # match_results = layer_utils.my_lstm_layer(VV_c_input, options.context_lstm_dim, scope_name="match_result", reuse=False,
        #                                     is_training=is_training, dropout_rate=dropout_rate)[2]
        # match_dim = 2 * options.context_lstm_dim
    # return (match_results, match_dim)
    return (VV_c_input, input_dim)

def onelayer_BiMPM_match(in_dim, passage, question, passage_mask, question_mask, accum_dim=0, passage_accum=None, question_accum=None,
               options=None, scope_name='onelayer_BiMPM_match', is_training=True, dropout_rate=0.2, reuse=False):
    if passage_accum is None:
        passage_accum = passage
        question_accum = question
        accum_dim = in_dim
    match_results = []
    match_dim = 0
    QoP_reps = None
    with tf.variable_scope(scope_name, reuse=reuse):
        # attention passage over question
        PoQ_atten = layer_utils.calcuate_attention(passage_accum, question_accum, accum_dim, accum_dim, scope_name="PoQ_atten",
                                                       att_type=options.att_type, att_dim=options.att_dim, remove_diagnoal=False,
                                                       mask1=passage_mask, mask2=question_mask,
                                                       is_training = is_training, dropout_rate = dropout_rate)
        PoQ_reps = tf.matmul(PoQ_atten, layer_utils.dropout_layer(question, dropout_rate, is_training=is_training))
        if options.with_QoP:
            # attention question over passage
            QoP_atten = layer_utils.calcuate_attention(question_accum, passage_accum, accum_dim, accum_dim, scope_name="QoP_atten",
                                                       att_type=options.att_type, att_dim=options.att_dim, remove_diagnoal=False,
                                                       mask1=question_mask, mask2=passage_mask,
                                                       is_training=is_training, dropout_rate=dropout_rate)
            QoP_reps = tf.matmul(QoP_atten, layer_utils.dropout_layer(passage, dropout_rate, is_training=is_training))

        # attentive matching
        (att_match_rep, att_match_dim) = multi_perspective_match(in_dim, passage, PoQ_reps,
                                    is_training=is_training, dropout_rate=dropout_rate, options=options, scope_name='att_match')
        match_results.append(att_match_rep)
        match_dim += att_match_dim

        # max attentive matching
        PoQ_max_reps = layer_utils.collect_representation(question, tf.argmax(PoQ_atten, axis=2, output_type=tf.int32))
        (max_att_match_rep, max_att_match_dim) = multi_perspective_match(in_dim, passage, PoQ_max_reps,
                                    is_training=is_training, dropout_rate=dropout_rate, options=options, scope_name='max_att_match')
        match_results.append(max_att_match_rep)
        match_dim += max_att_match_dim

    match_results = tf.concat(axis=2, values=match_results)
    return (match_results, match_dim, PoQ_reps, QoP_reps)


def BiMPM_match(feature_dim, passage, question, passage_length, question_length, passage_mask, question_mask, onehot_binary=None,
               options=None, scope_name='BiMPM_match_layer', is_training=True, dropout_rate=0.2, reuse=False):
    match_results = []
    match_dim = 0
    with tf.variable_scope(scope_name, reuse=reuse):
        # word-level matching
        (word_match_reps, word_match_dim, word_PoQ_reps, word_QoP_reps) = onelayer_BiMPM_match(feature_dim, passage, question,
                        passage_mask, question_mask, options=options, scope_name='word_level_BiMPM',
                        is_training=is_training, dropout_rate=dropout_rate, reuse=False)
        match_results.append(word_match_reps)
        match_dim += word_match_dim


        # contextual level matching
        passage_reps = [passage, word_PoQ_reps]
        passage_dim = 2 * feature_dim
        # if onehot_binary is not None:
        #     passage_reps.append(onehot_binary)
        #     passage_dim += 11

        question_reps = [question]
        if options.with_QoP: question_reps.append(word_QoP_reps)

        passage_context = passage
        if onehot_binary is not None:
            passage_context = tf.concat(axis=2, values=[passage_context, onehot_binary])
        question_context = question
        for i in xrange(options.context_layer_num):
            cur_passage_reps = tf.concat(axis=2, values=passage_reps)
            cur_question_reps = tf.concat(axis=2, values=question_reps)

            # lstm over passage and question individually
            passage_context = layer_utils.my_lstm_layer(passage_context, options.context_lstm_dim,
                scope_name="passage_context_lstm_{}".format(i), reuse=False, is_training=is_training, dropout_rate=dropout_rate)[2]
            passage_context = tf.multiply(passage_context, tf.expand_dims(passage_mask, axis=-1))
            question_context = layer_utils.my_lstm_layer(question_context, options.context_lstm_dim,
                scope_name="question_context_lstm_{}".format(i), reuse=False, is_training=is_training, dropout_rate=dropout_rate)[2]
            question_context = tf.multiply(question_context, tf.expand_dims(question_mask, axis=-1))

            # matching
            (cur_match_reps, cur_match_dim, cur_PoQ_reps, cur_QoP_reps) = onelayer_BiMPM_match(2*options.context_lstm_dim,
                                passage_context, question_context, passage_mask, question_mask,
                                accum_dim=passage_dim, passage_accum=cur_passage_reps, question_accum=cur_question_reps,options=options,
                                scope_name='context_BiMPM_{}'.format(i), is_training=is_training, dropout_rate=dropout_rate, reuse=False)

            match_results.append(cur_match_reps)
            match_dim += cur_match_dim

            if options.accumulate_match_input:
                passage_reps.append(passage_context)
                passage_reps.append(cur_PoQ_reps)
                # passage_reps.append(cur_match_reps)
                passage_dim += 4 * options.context_lstm_dim
                question_reps.append(question_context)
                if options.with_QoP: question_reps.append(cur_QoP_reps)
            else:
                # passage_reps = [passage_context, cur_PoQ_reps, cur_match_reps]
                passage_reps = [passage_context, cur_PoQ_reps]
                passage_dim = 4 * options.context_lstm_dim
                question_reps = [question_context]
                if options.with_QoP: question_reps.append(cur_QoP_reps)

        match_results = tf.concat(axis=2, values=match_results)
        if options.with_self_match:
            cur_passage_reps = tf.concat(axis=2, values=passage_reps)
            cur_passage_reps_projection = layer_utils.projection_layer(cur_passage_reps, passage_dim, options.self_compress_dim,
                                                                 scope="self-match-projection")
            self_atten_scores = layer_utils.calcuate_attention(cur_passage_reps_projection, cur_passage_reps_projection,
                                                               options.self_compress_dim, options.self_compress_dim,
                                                               scope_name="self_boost_att", att_type=options.att_type,
                                                               att_dim=options.att_dim, remove_diagnoal=True,
                                                               mask1=passage_mask, mask2=passage_mask,
                                                               is_training=is_training, dropout_rate=dropout_rate)
            self_match_reps = tf.matmul(self_atten_scores, layer_utils.dropout_layer(match_results, dropout_rate, is_training=is_training))
            match_results = tf.concat(axis=2, values=[match_results, self_match_reps])
            match_dim = 2 * match_dim
    return (match_results, match_dim)

def FusionNet_match_Amit(feature_dim, feature_each_dim, passage, question, passage_length, question_length,
                         passage_mask, question_mask, onehot_binary=None,
                         options=None, scope_name='FusionNet_Amit_match_layer',
                         is_training=True, dropout_rate=0.2, reuse=False):
    batch_size = tf.shape(passage)[0]
    passage_len = tf.shape(passage)[1]
    question_len = tf.shape(question)[1]
    word_dim, char_dim, POS_dim, NER_dim, cove_dim, lm_dim = feature_each_dim

    with tf.variable_scope(scope_name, reuse=reuse):
        # Fully Aware MultiLevel Fusion (FAMF) Word  Layer
        with tf.variable_scope('famf_word_layer'):
            famf_word_level_dim = word_dim  # assuming famf_word_level_dim=dim-of-glove=300
            p_wordlevel_input = tf.slice(passage,  [0, 0, 0], [batch_size, passage_len,  word_dim]) # only use word embedding for word layer
            q_wordlevel_input = tf.slice(question, [0, 0, 0], [batch_size, question_len, word_dim])
            alphas = layer_utils.calcuate_attention(p_wordlevel_input, q_wordlevel_input, famf_word_level_dim, famf_word_level_dim,
                                                    scope_name="famf_word_layer_attention",
                                                    att_type=options.att_type, mask1=passage_mask, mask2=question_mask,
                                                    att_dim=250, is_training=is_training, dropout_rate=dropout_rate)
            # (in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
            #            att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None, mask2=None, is_training=False, dropout_rate=0.2,
            #            cosine_attention_scale=200)
            weighted_by_question_words = tf.matmul(alphas, layer_utils.dropout_layer(q_wordlevel_input, dropout_rate, is_training=is_training))

        # Reading layer
        with tf.variable_scope('reading'):
            q_rep_reading_input = question  # [glove, cove, NER, POS]
            p_rep_reading_input = tf.concat(axis=2, values=[passage, onehot_binary, weighted_by_question_words]) # use all embeddings for reading and understanding.
            # [glove, cove, NER, POS, binary,famf_word_attention]

            with tf.variable_scope('reading_layer_1'):
                reading_layer_lstm_dim = 125

                q_rep_reading_1_output = layer_utils.my_lstm_layer(
                    q_rep_reading_input, reading_layer_lstm_dim, scope_name='bilstm_reading_1_q', reuse=False,
                    is_training=is_training, dropout_rate=options.dropout_rate)[2] # [B, Q, 250 ]

                p_rep_reading_1_output = layer_utils.my_lstm_layer(
                    p_rep_reading_input, reading_layer_lstm_dim, scope_name='bilstm_reading_1_p', reuse=False,
                    is_training=is_training, dropout_rate=options.dropout_rate)[2] # [B, Q, 250 ]

            with tf.variable_scope('reading_layer_2'):
                q_rep_reading_2_output = layer_utils.my_lstm_layer(
                    q_rep_reading_1_output, reading_layer_lstm_dim, scope_name='bilstm_reading_1_q', reuse=False,
                    is_training=is_training, dropout_rate=options.dropout_rate)[2]  # [B, Q, 250 ]

                p_rep_reading_2_output = layer_utils.my_lstm_layer(
                    p_rep_reading_1_output, reading_layer_lstm_dim, scope_name='bilstm_reading_1_p', reuse=False,
                    is_training=is_training, dropout_rate=options.dropout_rate)[2]  # [B, Q, 250 ]

        # Understanding Layer
        with tf.variable_scope('question_understanding_layer'):
            q_rep_understanding_input = tf.concat(axis=2, values=(q_rep_reading_1_output, q_rep_reading_2_output))
            U_q = layer_utils.my_lstm_layer(
                    q_rep_understanding_input, reading_layer_lstm_dim, scope_name='bilstm_understanding_q', reuse=False,
                    is_training=is_training, dropout_rate=options.dropout_rate)[2]  # [B, Q, 250 ]

            U_q_dim = reading_layer_lstm_dim * 2

        # FAMF : higher level
        with tf.variable_scope('famf_higher_layer'):
            famf_higher_layer_w_dim1 = 500
            famf_higher_layer_w_dim2 = 250
            famf_q_input = []
            famf_p_input = []

            # famf_p_input.append(in_passage_word_repres)
            famf_p_input.append(p_wordlevel_input)
            famf_higher_layer_w_dim1 += word_dim
            famf_p_input.append(p_rep_reading_1_output)
            famf_p_input.append(p_rep_reading_2_output)

            # famf_q_input.append(in_question_word_repres)
            famf_q_input.append(q_wordlevel_input)
            famf_q_input.append(q_rep_reading_1_output)
            famf_q_input.append(q_rep_reading_2_output)

            cove_dim_begin = word_dim + char_dim + POS_dim + NER_dim
            if cove_dim != 0:
                #cove_dim_begin = word_dim + char_dim + POS_dim + NER_dim
                p_cove_repres = tf.slice(passage, [0, 0, cove_dim_begin], [batch_size, passage_len, cove_dim])
                q_cove_repres = tf.slice(question, [0, 0, cove_dim_begin], [batch_size, question_len, cove_dim])
                famf_p_input.append(p_cove_repres)
                famf_q_input.append(q_cove_repres)
                famf_higher_layer_w_dim1 += cove_dim

            if lm_dim != 0:
                lm_dim_begin = cove_dim_begin + cove_dim
                p_lm_repres = tf.slice(passage, [0, 0, lm_dim_begin], [batch_size, passage_len, lm_dim])
                q_lm_repres = tf.slice(question, [0, 0, lm_dim_begin], [batch_size, question_len, lm_dim])
                famf_p_input.append(p_lm_repres)
                famf_q_input.append(q_lm_repres)
                famf_higher_layer_w_dim1 += lm_dim

            famf_p_input = tf.concat(axis=2, values=famf_p_input)  # (B, P, D )
            famf_q_input = tf.concat(axis=2, values=famf_q_input)  # (B, Q, D )

            alphas = layer_utils.calcuate_attention(famf_p_input, famf_q_input, famf_higher_layer_w_dim1, famf_higher_layer_w_dim1,
                                                    scope_name="famf_high_lowlevel",
                                                    att_type=options.att_type, mask1=passage_mask, mask2=question_mask,
                                                    att_dim=famf_higher_layer_w_dim2, is_training=is_training, dropout_rate=dropout_rate)
            h_Cl = tf.matmul(alphas, layer_utils.dropout_layer(q_rep_reading_1_output, dropout_rate, is_training=is_training))

            alphas = layer_utils.calcuate_attention(famf_p_input, famf_q_input, famf_higher_layer_w_dim1, famf_higher_layer_w_dim1,
                                                    scope_name="famf_high_highlevel",
                                                    att_type=options.att_type, mask1=passage_mask, mask2=question_mask,
                                                    att_dim=famf_higher_layer_w_dim2, is_training=is_training, dropout_rate=dropout_rate)
            h_Ch = tf.matmul(alphas, layer_utils.dropout_layer(q_rep_reading_2_output, dropout_rate, is_training=is_training))

            alphas = layer_utils.calcuate_attention(famf_p_input, famf_q_input, famf_higher_layer_w_dim1, famf_higher_layer_w_dim1,
                                                    scope_name="famf_high_understandinglevel",
                                                    att_type=options.att_type, mask1=passage_mask, mask2=question_mask,
                                                    att_dim=famf_higher_layer_w_dim2, is_training=is_training, dropout_rate=dropout_rate)
            u_C = tf.matmul(alphas, layer_utils.dropout_layer(U_q, dropout_rate, is_training=is_training))


            with tf.variable_scope('famf_higher_layer_passage_lstm'):
                p_rep_highlayer_input = []
                p_rep_highlayer_input.append(p_rep_reading_1_output)
                p_rep_highlayer_input.append(p_rep_reading_2_output)
                p_rep_highlayer_input.append(h_Cl)
                p_rep_highlayer_input.append(h_Ch)
                p_rep_highlayer_input.append(u_C)
                p_rep_highlayer_input = tf.concat(axis=2, values=p_rep_highlayer_input)  # (B, P, D ) D=(250*5)

                famf_higher_layer_passage_lstm_dim = 125

                V_c = layer_utils.my_lstm_layer(
                    p_rep_highlayer_input, famf_higher_layer_passage_lstm_dim, scope_name='bilstm_higher_layer_p', reuse=False,
                   is_training=is_training, dropout_rate=options.dropout_rate)[2]  # [B, Q, 250 ]


        # FAMF: Self-boosted
        with tf.variable_scope('famf_selfboosted_layer'):
            famf_self_boosted_input = []
            famf_self_boosted_w_dim1 = 250 * 6

            # famf_self_boosted_input.append(in_passage_word_repres)
            famf_self_boosted_input.append(p_wordlevel_input)
            famf_self_boosted_w_dim1 += word_dim
            famf_self_boosted_input.append(p_rep_reading_1_output)
            famf_self_boosted_input.append(p_rep_reading_2_output)
            famf_self_boosted_input.append(h_Cl)
            famf_self_boosted_input.append(h_Ch)
            famf_self_boosted_input.append(u_C)
            famf_self_boosted_input.append(V_c)

            if cove_dim != 0:
                famf_self_boosted_input.append(tf.slice(passage, [0, 0, cove_dim_begin], [batch_size, passage_len, cove_dim]))
                famf_self_boosted_w_dim1 += cove_dim   # 300 + (250 * 6) + 600(if cove) + 300 (if lm)

            # if lm_dim != 0: not used in old codebase
            famf_self_boosted_w_dim2 = 50  # 250 does not fit in memory

            famf_self_boosted_input = tf.concat(axis=2,
                                                values=famf_self_boosted_input)  # (B, P, D ) D=(600 ,300 , 250*6 ) = 2400

            useProjectionLayer = True
            if useProjectionLayer:
                projection_dim = 50
                famf_self_boosted_input_dropout = famf_self_boosted_input
                famf_self_boosted_projection = layer_utils.projection_layer(famf_self_boosted_input_dropout, famf_self_boosted_w_dim1,
                                                                            projection_dim, scope="self-match-projection")
                famf_self_boosted_w_dim1 = projection_dim
                vv_C_input = famf_self_boosted_projection
            else:
                vv_C_input = famf_self_boosted_input

            alphas = layer_utils.calcuate_attention(vv_C_input, vv_C_input, famf_self_boosted_w_dim1, famf_self_boosted_w_dim1,
                                                    scope_name="famf_selfboosted_layer_attention",
                                                    att_type=options.att_type, mask1=passage_mask, mask2=passage_mask,
                                                    att_dim=famf_self_boosted_w_dim2, is_training=is_training, dropout_rate=dropout_rate)
            vv_C = tf.matmul(alphas, layer_utils.dropout_layer(V_c, dropout_rate, is_training=is_training))

            p_rep_selfboosted_layer_input = tf.concat(axis=2, values=(famf_self_boosted_input, vv_C))
            return (p_rep_selfboosted_layer_input, 0)




