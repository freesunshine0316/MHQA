import codecs
import json
import re
import sys
import numpy as np
import random
import torch
import utils

from allennlp.modules.elmo import Elmo, batch_to_ids

def span_distance(sp1, sp2):
    sp1, sp2 = sorted([sp1,sp2])
    if sp1[0] <= sp2[0] <= sp1[1] or sp1[0] <= sp2[1] <= sp1[1]:
        return 0
    assert sp1[1] < sp2[0], '{} {}'.format(sp1, sp2)
    return sp2[0] - sp1[1]


def make_edges(mentions, mentions_str, corefs, corefs_dict, stopword_set, options):
    mention_max_num = len(mentions)
    edges = [set() for x in mentions]
    for i in range(len(mentions)):
        if options.with_coref and i in corefs_dict:
            for j in corefs[corefs_dict[i]]:
                assert j < mention_max_num
                edges[j].add(i)
                edges[i].add(j)
        for j in range(i):
            if options.with_window and span_distance(mentions[i], mentions[j]) <= options.distance_thres:
                edges[j].add(i)
                edges[i].add(j)
            if options.with_same and mentions_str[i] == mentions_str[j] and \
                    mentions_str[i] not in stopword_set and \
                    span_distance(mentions[i], mentions[j]) > options.distance_thres_upper:
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
    all_instances = []
    miss_cand_count = 0
    miss_ref_count = 0
    total_count = 0
    stopword_set = {'of', 'in', 'a', 'it', 'he', 'she', 'his', 'her'}
    entity_type_map = {'none':0, 'query':1, 'standard':2, 'candidate':3}
    all_data = json.load(open(inpath, 'r'))
    for i, inst in enumerate(all_data):
        if subset_ids != None and inst['id'] not in subset_ids:
            continue
        if options.max_passage_size < sum(len(x['toks']) \
                for x in inst['supports_anno']):
            thres = int(options.max_passage_size / len(inst['supports_anno']))
        else:
            thres = -1

        # process documents
        inst_p_tok = []
        passage_lens = []
        inst_m_type = []
        mentions = [] # (st,ed)
        mentions_str = []
        mentions_dict = {} # (st,ed) --> id
        coref_clusters = [] # list of sets
        coref_clusters_dict = {}
        for j, web_snippet in enumerate(inst['supports_anno']): # for each passage
            # tokens
            cur_p_tok = web_snippet['toks']
            if thres > 0:
                cur_p_tok = cur_p_tok[:thres]
            passage_lens.append(len(cur_p_tok))
            prev_p_size = len(inst_p_tok)
            inst_p_tok += cur_p_tok
            # named entities
            for k, (ori_st, ori_ed) in enumerate(web_snippet['entities']):
                ori_ed -= 1
                assert ori_st <= ori_ed
                if thres > 0 and ori_ed >= len(cur_p_tok): # the mention has been cut
                    continue
                else:
                    assert ori_ed < len(cur_p_tok)
                st, ed = ori_st + prev_p_size, ori_ed + prev_p_size
                if (st,ed) not in mentions_dict:
                    inst_m_type.append(entity_type_map['standard'])
                    mentions.append((st,ed))
                    mentions_str.append(' '.join(inst_p_tok[st:ed+1]))
                    mentions_dict[(st,ed)] = len(mentions) - 1
            # coref clusters
            for cluster in web_snippet['coref_clusters']:
                cur_cluster = set()
                for ori_st, ori_ed in cluster:
                    # from my observation, coref annotations are "[]", not "[)"
                    assert ori_st <= ori_ed
                    if thres > 0 and ori_ed >= len(cur_p_tok): # the coref-mention has been cut
                        continue
                    else:
                        assert ori_ed < len(cur_p_tok)
                    st, ed = ori_st + prev_p_size, ori_ed + prev_p_size
                    if (st,ed) not in mentions_dict:
                        inst_m_type.append(entity_type_map['standard'])
                        mentions.append((st,ed))
                        mentions_str.append(' '.join(inst_p_tok[st:ed+1]))
                        mentions_dict[(st,ed)] = len(mentions) - 1
                    mid = mentions_dict[(st,ed)]
                    cur_cluster.add(mid)
                    coref_clusters_dict[mid] = len(coref_clusters)
                coref_clusters.append(cur_cluster)

        # process query
        for j, ori_st, ori_ed in inst['query_anno']['poses']:
            ori_ed -= 1
            assert ori_st <= ori_ed
            if thres > 0 and ori_ed >= passage_lens[j]:
                continue
            else:
                assert ori_ed < passage_lens[j]
            prev_p_size = sum(passage_lens[:j])
            st, ed = ori_st + prev_p_size, ori_ed + prev_p_size
            if (st,ed) not in mentions_dict:
                inst_m_type.append(entity_type_map['query'])
                mentions.append((st,ed))
                mentions_str.append(' '.join(inst_p_tok[st:ed+1]))
                mid = len(mentions) - 1
                mentions_dict[(st,ed)] = mid
            else:
                mid = mentions_dict[(st,ed)]
                inst_m_type[mid] = entity_type_map['query']
            qstr_1 = mentions_str[mid].lower()
            qstr_2 = ' '.join(inst['query_anno']['toks']).lower()
            assert qstr_2.endswith(qstr_1), '{} ||| {}'.format(qstr_1, qstr_2)

        # process candidates
        inst_c = [] # [cands, positions]
        for cid, cand in enumerate(inst['candidates_anno']):
            cand_poses = []
            for j, ori_st in cand['poses']:
                ori_ed = ori_st + len(cand['toks']) - 1
                assert ori_st <= ori_ed
                if thres > 0 and ori_ed >= passage_lens[j]:
                    continue
                else:
                    assert ori_ed < passage_lens[j]
                prev_p_size = sum(passage_lens[:j])
                st, ed = ori_st + prev_p_size, ori_ed + prev_p_size
                if (st,ed) not in mentions_dict:
                    inst_m_type.append(entity_type_map['candidate'])
                    mentions.append((st,ed))
                    mentions_str.append(' '.join(inst_p_tok[st:ed+1]))
                    mentions_dict[(st,ed)] = len(mentions) - 1
                cand_poses.append(mentions_dict[(st,ed)])
            inst_c.append(cand_poses)

        # CHECK candidate positions correctness
        assert len(inst_c) > 0
        for j, inst_c_item in enumerate(inst_c):
            given_str = ' '.join(inst['candidates_anno'][j]['toks']).lower()
            for mid in inst_c_item:
                passage_str = mentions_str[mid].lower()
                assert passage_str == given_str

        # process ref
        total_count += 1
        answer_cid = inst['answer_cid']
        # if ref not found in the candidate list
        if answer_cid is not None and answer_cid == -1:
            miss_ref_count += 1
            filtered_instances[inst['id']] = inst['candidates'][0] if 'candidates' in inst else 'None'
            continue
        # if the right candidate does not appear in the input
        if answer_cid is not None and len(inst_c[answer_cid]) == 0:
            miss_ref_count += 1
            filtered_instances[inst['id']] = inst['candidates'][0] if 'candidates' in inst else 'None'
            continue

        # remove candidates with zero appearances and update r_cid
        r_st, r_ed = mentions[inst_c[answer_cid][0]]
        r_str = inst_p_tok[r_st:r_ed]

        inst_c_nonempty = []
        inst_c_str = []
        for cid in range(len(inst_c)):
            if inst_c[cid] != []:
                inst_c_nonempty.append(inst_c[cid])
                inst_c_str.append(inst['candidates'][cid])
        inst_r_cid = None
        if answer_cid is not None:
            inst_r_cid = sum(inst_c[cid] != [] for cid in range(answer_cid))
        inst_c = inst_c_nonempty

        # CHECK ref position after move
        r_st, r_ed = mentions[inst_c[inst_r_cid][0]]
        assert r_str == inst_p_tok[r_st:r_ed]

        # make edges for the instance
        inst_eg = None
        if options.graph_encoding in ("GCN", "GRN"):
            inst_eg, _, _ = make_edges(mentions, mentions_str, coref_clusters, coref_clusters_dict, stopword_set, options)
        # add to the main data stream
        inst_m_st = [x[0] for x in mentions]
        inst_m_ed = [x[1] for x in mentions]
        all_instances.append({'question':inst['query_anno']['toks'], 'passage':inst_p_tok,
            'mention_starts':inst_m_st, 'mention_ends':inst_m_ed, 'mention_types': inst_m_type, 'edges':inst_eg,
            'candidates':inst_c, 'candidate_str':inst_c_str, 'ref':inst_r_cid, 'id':inst['id']})

    assert len(all_instances) + len(filtered_instances) == total_count
    print('Total {}, missing all candidates {}, missing answer {}'.format(total_count, miss_cand_count, miss_ref_count))
    return all_instances, filtered_instances


def make_batches_elmo(features, config, is_sort=True, is_shuffle=False):
    if is_sort:
        features.sort(key=lambda x: len(x['passage']))
    elif is_shuffle:
        random.shuffle(features)

    has_ref = features[0]['ref'] is not None

    N = 0
    batches = []
    while N < len(features):
        B = min(config.batch_size, len(features) - N)

        # useless under ELMo
        #input_lens = []
        #maxlen = 0
        #for i in range(B):
        #    l = len(features[N+i]['passage'])
        #    input_lens.append(l)
        #    maxlen = max(maxlen, l)
        #input_lens = torch.tensor(input_lens, dtype=torch.long)

        # textual input
        passage_ids = batch_to_ids([features[N+i]['passage'] for i in range(0, B)]) # [batch, passage]
        passage_lens = torch.tensor([len(features[N+i]['passage']) for i in range(0, B)], dtype=torch.long) # [batch]
        _, passage_max_len, other = list(passage_ids.size())
        extra_len = 10 - (passage_max_len % 10)
        if extra_len < 10:
            extra_ids = torch.zeros(B, extra_len, other, dtype=torch.long)
            passage_ids = torch.cat([passage_ids, extra_ids], dim=1)
        question_ids = batch_to_ids([features[N+i]['question'] for i in range(0, B)]) # [batch, question]
        question_lens = torch.tensor([len(features[N+i]['question']) for i in range(0, B)], dtype=torch.long) # [batch]
        assert (question_lens > 0).all()

        # mention
        mention_nums = []
        edge_nums_lst = [] if config.graph_encoding in ("GCN", "GRN") else None
        maxmnum = 0
        maxegnum = 0
        for i in range(B):
            l = len(features[N+i]['mention_starts'])
            mention_nums.append(l)
            maxmnum = max(maxmnum, l)
            if config.graph_encoding in ("GCN", "GRN"):
                edge_nums_lst.append([])
                for j in range(l):
                    ll = len(features[N+i]['edges'][j])
                    edge_nums_lst[-1].append(ll)
                    maxegnum = max(maxegnum, ll)
        mention_nums = torch.tensor(mention_nums, dtype=torch.long)
        assert (mention_nums > 0).all()

        mention_starts = np.zeros([B, maxmnum], dtype=np.long)
        mention_ends = np.zeros([B, maxmnum], dtype=np.long)
        mention_types = np.zeros([B, maxmnum], dtype=np.long)
        edge_nums = None
        if config.graph_encoding in ("GCN", "GRN"):
            edge_nums = np.zeros([B, maxmnum], dtype=np.long)
        for i in range(B):
            curseq = len(features[N+i]['mention_starts'])
            mention_starts[i,:curseq] = features[N+i]['mention_starts']
            mention_ends[i,:curseq] = features[N+i]['mention_ends']
            mention_types[i,:curseq] = features[N+i]['mention_types']
            if config.graph_encoding in ("GCN", "GRN"):
                edge_nums[i,:curseq] = edge_nums_lst[i]
        mention_starts = torch.tensor(mention_starts, dtype=torch.long)
        mention_ends = torch.tensor(mention_ends, dtype=torch.long)
        mention_types = torch.tensor(mention_types, dtype=torch.long)
        if config.graph_encoding in ("GCN", "GRN"):
            edge_nums = torch.tensor(edge_nums, dtype=torch.long)

        edges = None
        if config.graph_encoding in ("GCN", "GRN"):
            edges = np.zeros([B, maxmnum, maxegnum], dtype=np.long)
            for i in range(B):
                for j in range(mention_nums[i]):
                    curseq = edge_nums_lst[i][j]
                    edges[i,j,:curseq] = features[N+i]['edges'][j]
            edges = torch.tensor(edges, dtype=torch.long)
            assert (edges < mention_nums.view(B, 1, 1)).all()

        # candidate and reference
        candidate_num = []
        candidate_appear_num_lst = []
        maxcnum = 0
        maxcpnum = 0
        for i in range(B):
            l = len(features[N+i]['candidates'])
            candidate_num.append(l)
            maxcnum = max(maxcnum, l)
            candidate_appear_num_lst.append([])
            for j in range(l):
                ll = len(features[N+i]['candidates'][j])
                candidate_appear_num_lst[-1].append(ll)
                maxcpnum = max(maxcpnum, ll)
        candidate_num = torch.tensor(candidate_num, dtype=torch.long)
        assert (candidate_num > 0).all()

        candidate_appear_num = np.zeros([B, maxcnum], dtype=np.long)
        for i in range(B):
            curseq = candidate_num[i]
            candidate_appear_num[i,:curseq] = candidate_appear_num_lst[i]
        candidate_appear_num = torch.tensor(candidate_appear_num, dtype=torch.long)

        candidate_str = [features[N+i]['candidate_str'] for i in range(0, B)]
        candidates = np.zeros([B, maxcnum, maxcpnum], dtype=np.long)
        for i in range(B):
            for j in range(candidate_num[i]):
                curseq = candidate_appear_num_lst[i][j]
                candidates[i,j,:curseq] = features[N+i]['candidates'][j]
        candidates = torch.tensor(candidates, dtype=torch.long)

        refs = None
        if has_ref:
            refs = torch.tensor([features[N+i]['ref'] for i in range(0, B)], dtype=torch.long)

        ids = [features[N+i]['id'] for i in range(0, B)]

        batches.append({'passage_ids':passage_ids, 'passage_lens':passage_lens, 'question_ids':question_ids,
            'question_lens': question_lens, 'ids':ids, 'refs':refs,
            'mention_nums':mention_nums, 'mention_starts':mention_starts, 'mention_ends':mention_ends,
            'mention_types':mention_types, 'edge_nums':edge_nums, 'edges':edges, 'candidate_num':candidate_num,
            'candidate_appear_num':candidate_appear_num, 'candidates':candidates, 'candidate_str':candidate_str})
        N += B
    return batches



