
import os, sys, json
import torch
import numpy as np

import spacy
spacy.require_gpu()
nlp = spacy.load("en_core_web_sm")

#import neuralcoref
#neuralcoref.add_to_pipe(nlp)

from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz",
        cuda_device=2)

def search_list(sublst, lst):
    starts = []
    for i in range(len(lst) - len(sublst) + 1):
        if lst[i:i+len(sublst)] == sublst:
            starts.append(i)
    return starts


n = 0
data = json.load(open(sys.argv[1], 'r'))
for instance in data:
    # passage annotation
    if 'supports_anno' not in instance:
        instance['supports_anno'] = []
        doc_toks_lower = []
        for doc_raw in instance['supports']:
            doc = nlp(doc_raw)
            toks = [t.text for t in doc]
            doc_toks_lower.append([t.text.lower() for t in doc])
            entities = [(ne.start, ne.end) for ne in doc.ents]
            coref_clusters = predictor.predict_tokenized(toks)['clusters'] # list of list of span
            sentence_starts = [snt.start for snt in doc.sents]
            sentence_ends = [snt.end for snt in doc.sents]
            instance['supports_anno'].append({'toks':toks, 'entities':entities, 'coref_clusters':coref_clusters,
                'sentence_starts':sentence_starts, 'sentence_ends':sentence_ends})
    # question annotation
    if 'query_anno' not in instance:
        query = instance['query'].split()
        query_subj = query[1:]
        query_subj_len = len(query_subj)
        query_poses = []
        if query_subj_len > 0:
            for i, doc in enumerate(instance['supports_anno']):
                doc_toks_lower = [x.lower() for x in doc['toks']]
                query_poses.extend((i,j,j+query_subj_len) for j in search_list(query_subj, doc_toks_lower))
        instance['query_anno'] = {'toks': query[0].split('_') + query_subj, 'poses':query_poses}
    # candidate annotation
    if 'candidates_anno' not in instance:
        instance['candidates_anno'] = []
        is_cand_miss = []
        for cand in instance['candidates']:
            cand_toks = [t.text for t in nlp(cand)]
            cand_poses = []
            for i in range(len(instance['supports'])):
                cand_poses.extend((i,j) for j in search_list(cand_toks, doc_toks_lower[i]))
            is_cand_miss.append(len(cand_poses) == 0)
            instance['candidates_anno'].append({'toks':cand_toks, 'poses':cand_poses})
    # answer annotation
    if 'answer' not in instance:
        instance['answer_cid'] = None
    elif 'answer_cid' not in instance:
        ans_toks = [t.text for t in nlp(instance['answer'])]
        instance['answer_cid'] = -1
        for i, cand in enumerate(instance['candidates_anno']):
            if cand['toks'] == ans_toks:
                instance['answer_cid'] = i
                break
    n += 1
    if n % 100 == 0:
        print(n)

    # final check
    if 'is_cand_miss' not in locals():
        continue

    if instance['answer_cid'] is None:
        missed_cand = [x for i, x in enumerate(instance['candidates']) if is_cand_miss[i]]
        print("!!!Final test !!!At least one cand missed: {}".format(missed_cand))
        continue

    if instance['answer_cid'] == -1:
        print("!!!Answer is not found in candidates: {}, {}".format(instance['answer'], instance['candidates']))
    elif np.all(is_cand_miss):
        print("!!!All cand missed: {}".format(instance['candidates']))
    elif instance['answer_cid'] >= 0 and is_cand_miss[instance['answer_cid']]:
        missed_cand = ['@@'+x if i == instance['answer_cid'] else x \
                for i, x in enumerate(instance['candidates']) if is_cand_miss[i]]
        print("!!!Answer cand missed: {}".format(missed_cand))
    elif np.any(is_cand_miss):
        missed_cand = ['@@'+x if i == instance['answer_cid'] else x \
                for i, x in enumerate(instance['candidates']) if is_cand_miss[i]]
        print("!!!At least one cand missed: {}".format(missed_cand))

json.dump(data, open(sys.argv[1]+'_prep','w'))
