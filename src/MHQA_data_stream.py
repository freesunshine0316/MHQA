import codecs
import json
import re
import sys
import numpy as np
import random
import padding_utils


def make_edges(mentions, mentions_region, mentions_str, corefs, stopword_set, options):
    edges = [set() for x in mentions]
    for i in range(len(mentions)):
        if options.with_coref and i in corefs:
            coref_i = corefs[i]
            edges[coref_i].add(i)
            edges[i].add(coref_i)
        for j in range(i):
            if options.with_window and mentions_region[j] == mentions_region[i] and \
                    min(abs(mentions[i][0] - mentions[j][1]),
                            abs(mentions[j][0] - mentions[i][1])) <= options.distance_thres:
                edges[j].add(i)
                edges[i].add(j)
            if options.with_same and mentions_str[i] == mentions_str[j] and \
                    mentions_str[i] not in stopword_set and \
                    (mentions_region[i] != mentions_region[j] or
                        min(abs(mentions[i][0] - mentions[j][1]),
                            abs(mentions[j][0] - mentions[i][1])) > options.distance_thres_upper):
                edges[j].add(i)
                edges[i].add(j)
    max_edges = max(len(x) for x in edges)
    sum_edges = sum(len(x) for x in edges)
    if max_edges > options.max_edges:
        print('!!!{}'.format(max_edges))
    edges = [list(x)[:options.max_edges] for x in edges]
    return edges, max_edges, sum_edges


def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines


def read_data_file(inpath, options, subset_ids=None):
    filtered_instances = {}
    passage = [] # [batch, passage_len]
    question = [] # [batch, question_len]
    entity_start = [] # [batch, entity_num]
    entity_end = [] # [batch, entity_num]
    edges = [] # [batch, entity_num, edge_num]
    ids = []
    candidates = [] # [batch, cand_num, cand_occu_num]
    candidates_str = [] # str [batch, cand_num]
    ref = [] # [batch]
    miss_cand_count = 0
    miss_ref_count = 0
    total_count = 0
    stopword_set = set(['of', 'in', 'a', 'it', 'he', 'she', 'his', 'her'])
    with codecs.open(inpath, "rU", "utf-8") as f:
        for i, line in enumerate(f):
            inst = json.loads(line.strip())
            if subset_ids != None and inst['id'] not in subset_ids:
                continue
            if options.max_passage_size < sum(len(x['annotations']['toks'].split()) \
                    for x in inst['web_snippets']):
                thres = options.max_passage_size / len(inst['web_snippets'])
            else:
                thres = -1
            inst_p_tok = []
            inst_e_st = []
            inst_e_ed = []
            mentions = [] # (st,ed)
            mentions_dict = {} # (st,ed) --> id
            mentions_position_dict = {} # a:b --> id
            mentions_region = [] # which passage it belongs to
            mentions_str = []
            corefs = {} # id (tgt) --> id (ori)
            for j, web_snippet in enumerate(inst['web_snippets']): # for each passage
                cur_p_tok = web_snippet['annotations']['toks'].lower().split()
                if thres > 0:
                    cur_p_tok = cur_p_tok[:thres]
                prev_p_size = len(inst_p_tok)
                inst_p_tok += cur_p_tok
                for k, x in enumerate(web_snippet['annotations']['mentions'].split()):
                    st, ed = x.split('-')
                    st, ed = int(st)+prev_p_size, int(ed)+1+prev_p_size
                    if ed > len(inst_p_tok): # the mention has been cut
                        continue
                    inst_e_st.append(st)
                    inst_e_ed.append(ed)
                    mentions.append((st,ed))
                    mentions_dict[(st,ed)] = len(mentions)-1
                    mentions_position_dict['%d:%d'%(j,k)] = len(mentions)-1
                    mentions_region.append(j)
                    mentions_str.append(' '.join(inst_p_tok[st:ed]))
                    assert cur_p_tok[st-prev_p_size:ed-prev_p_size] == inst_p_tok[st:ed] # make sure the offset change is right
                for x in web_snippet['annotations']['corefs']: # corefs
                    if x == '': continue
                    cluster = []
                    for y in x.split():
                        st, ed, hd = y.split('-')
                        st, ed, hd = int(st)+prev_p_size, int(ed)+1+prev_p_size, int(hd)+prev_p_size
                        if ed > len(inst_p_tok): # the coref has been cut
                            continue
                        cluster.append(mentions_dict[(st,ed)])
                    cluster = sorted(cluster)
                    for k in range(1,len(cluster)):
                        corefs[cluster[k]] = cluster[0]
            # make edges for the entire instance
            if options.with_grn or options.with_gcn:
                inst_edges, _, _ = make_edges(mentions, mentions_region, mentions_str, corefs, stopword_set, options)
            else:
                inst_edges = None
            # process candidates & answer
            total_count += 1
            inst_c = [[mentions_position_dict.get(y,None) for y in x.split()] for x in inst['candidates']]
            inst_c = [[y for y in x if y != None] for x in inst_c]
            if all(len(x) == 0 for x in inst_c):
                miss_cand_count += 1
                filtered_instances[inst['id']] = \
                        inst['candidates_str'][0] if 'candidates_str' in inst else 'None'
                continue
            inst_r_idx = inst['answer'] if 'answer' in inst else -1
            assert inst_r_idx < len(inst_c)
            # works only on Dev and Train sets (since they have refs)
            if inst_r_idx >= 0 and len(inst_c[inst_r_idx]) == 0:
                miss_ref_count += 1
                filtered_instances[inst['id']] = \
                        inst['candidates_str'][0] if 'candidates_str' in inst else 'None'
                continue
            if inst_r_idx >= 0:
                inst_r_idx -= sum(len(x) == 0 for x in inst_c[:inst_r_idx])
            inst_c = [x for x in inst_c if len(x) > 0]
            inst_c_str = [' '.join(inst_p_tok[mentions[x[0]][0]: mentions[x[0]][1]]) for x in inst_c]
            assert len(inst_c_str) > 0
            # add to the main data stream
            question.append(inst['annotations']['toks'].lower().split())
            passage.append(inst_p_tok)
            entity_start.append(inst_e_st)
            entity_end.append(inst_e_ed)
            edges.append(inst_edges)
            candidates.append(inst_c)
            candidates_str.append(inst_c_str)
            ref.append(inst_r_idx)
            ids.append(inst['id'])

    assert len(ids) + len(filtered_instances) == total_count
    if sum(x == -1 for x in ref) != len(ref) and sum(x >= 0 for x in ref) != len(ref):
        assert False, '!!! Error, only some instances have reference !!!'
    has_ref = ref[0] >= 0
    print('Total {}, missing all candidates {}, missing answer {}'.format(total_count, miss_cand_count, miss_ref_count))
    return zip(question, passage, entity_start, entity_end, edges, candidates, ref, ids,
            candidates_str), filtered_instances, has_ref


def collect_vocabs(all_instances):
    all_words = set()
    all_chars = set()
    for (question, passage, entity_start, entity_end, edges, candidates, ref, ids,
            candidates_str) in all_instances:
        all_words.update(question)
        all_words.update(passage)
    for w in all_words:
        all_chars.update(w)
    return (all_words, all_chars)


class DataStream(object):
    def __init__(self, all_instances, word_vocab=None, char_vocab=None, options=None,
                 isShuffle=False, isLoop=False, isSort=True, has_ref=True, batch_size=-1):
        self.options = options
        if batch_size ==-1: batch_size=options.batch_size
        # index tokens and filter the dataset
        instances = []
        for (question, passage, entity_start, entity_end, edges, candidates, ref, ids,
                candidates_str) in all_instances:
            question_idx = word_vocab.to_index_sequence_for_list(question)
            passage_idx = word_vocab.to_index_sequence_for_list(passage)
            question_chars_idx = None
            passage_chars_idx = None
            if options.with_char:
                question_chars_idx = char_vocab.to_character_matrix_for_list(question,
                        max_char_per_word=options.max_char_per_word)
                passage_chars_idx = char_vocab.to_character_matrix_for_list(question,
                        max_char_per_word=options.max_char_per_word)
            instances.append((question_idx, question_chars_idx,
                passage_idx, passage_chars_idx, entity_start, entity_end, edges, candidates, ref, ids,
                candidates_str))

        all_instances = instances
        instances = None

        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda inst: len(inst[2]))

        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in xrange(batch_start, batch_end):
                cur_instances.append(all_instances[i])
            cur_batch = Batch(cur_instances, options, word_vocab=word_vocab, has_ref=has_ref)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]

class Batch(object):
    def __init__(self, instances, options, word_vocab=None, char_vocab=None, has_ref=True):
        self.options = options

        self.batch_size = len(instances)
        self.options = options
        self.vocab = word_vocab
        self.char_vocab = char_vocab
        self.ids = [x[9] for x in instances] # [batch]
        self.candidates_str = [x[10] for x in instances] # [batch, cand_num]
        assert type(instances[0][8]) == int
        if has_ref:
            self.ref_str = [self.candidates_str[i][x[8]] for i, x in enumerate(instances)]
        else:
            self.ref_str = None

        # data
        self.question = [x[0] for x in instances] # [batch, qlen]
        self.passage = [x[2] for x in instances] # [batch, plen]
        self.entity_start = [x[4] for x in instances] # [batch, entity_num]
        self.entity_end = [x[5] for x in instances] # [batch, entity_num]
        if options.with_grn or options.with_gcn:
            self.entity_edges = [x[6] for x in instances] # [batch, entity_num, edge_num]
        else:
            self.entity_edges = None
        self.cands = [x[7] for x in instances] # [batch, cand_num, cand_occu_num]
        if has_ref:
            self.ref = [x[8] for x in instances] # [batch]
        else:
            self.ref = None
        if options.with_char:
            self.question_chars = [x[1] for x in instances] # [batch, qlen, char_len]
            self.passage_chars = [x[3] for x in instances] # [batch, plen, char_len]

        # create length
        self.question_len = [len(x) for x in self.question] # [batch_size]
        self.passage_len = [len(x) for x in self.passage] # [batch_size]
        self.entity_len = [len(x) for x in self.entity_start] # [batch_size]
        if options.with_grn or options.with_gcn:
            self.entity_edges_mask = [[[1 for y in x] for x in edges] for edges in self.entity_edges] # [batch_size, entity_num, edge_num]
        else:
            self.entity_edges_mask = None
        self.cands_len = [len(cands) for cands in self.cands] # [batch_size]
        self.cands_occur_mask = [[[1 for y in x] for x in cands] for cands in self.cands] # [batch_size, cand_num, cand_occu_num]
        if options.with_char:
            self.question_chars_num = [[len(x) for x in qchars] for qchars in self.question_chars] # [batch_size, qlen]
            self.passage_chars_num = [[len(x) for x in qchars] for qchars in self.passage_chars] # [batch_size,  plen]


class BatchPadded:
    def __init__(self, ori_batch):
        self.batch_size = ori_batch.batch_size
        self.options = ori_batch.options
        self.vocab = ori_batch.vocab
        self.char_vocab = ori_batch.char_vocab

        self.ids = ori_batch.ids
        self.candidates_str = ori_batch.candidates_str
        self.ref_str = ori_batch.ref_str

        # making ndarray
        self.question = padding_utils.pad_2d_vals_no_size(ori_batch.question)
        self.question_len = np.array(ori_batch.question_len, dtype=np.int32)
        self.passage = padding_utils.pad_2d_vals_no_size(ori_batch.passage)
        self.passage_len = np.array(ori_batch.passage_len, dtype=np.int32)
        self.entity_start = padding_utils.pad_2d_vals_no_size(ori_batch.entity_start)
        self.entity_end = padding_utils.pad_2d_vals_no_size(ori_batch.entity_end)
        self.entity_len = np.array(ori_batch.entity_len, dtype=np.int32)
        if self.options.with_grn or self.options.with_gcn:
            self.entity_edges = padding_utils.pad_3d_vals_no_size(ori_batch.entity_edges)
            self.entity_edges_mask = padding_utils.pad_3d_vals_no_size(ori_batch.entity_edges_mask, dtype=np.float32)
        else:
            self.entity_edges = None
            self.entity_edges_mask = None
        self.cands = padding_utils.pad_3d_vals_no_size(ori_batch.cands)
        self.cands_len = np.array(ori_batch.cands_len, dtype=np.int32)
        self.cands_occur_mask = padding_utils.pad_3d_vals_no_size(ori_batch.cands_occur_mask, dtype=np.float32)
        if ori_batch.ref != None:
            self.ref = np.array(ori_batch.ref, dtype=np.int32)
        else:
            self.ref = None
        if self.options.with_char:
            self.question_chars = padding_utils.pad_3d_vals_no_size(ori_batch.question_chars)
            self.question_chars_num = padding_utils.pad_2d_vals_no_size(ori_batch.question_chars_num)
            self.passage_chars = padding_utils.pad_3d_vals_no_size(ori_batch.passage_chars)
            self.passage_chars_num = padding_utils.pad_2d_vals_no_size(ori_batch.passage_chars_num)


class OPT:
    def __init__(self):
        self.distance_thres = 30
        self.max_passage_size = 100000

if __name__ == "__main__":
    read_data_file('../../data.wikihop/dev.compQA.json.with_annotation.slfprop', OPT())


