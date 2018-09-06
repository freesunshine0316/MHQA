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

import MHQA_data_stream
import namespace_utils

if __name__ == '__main__':
    # load the configuration file
    FLAGS = namespace_utils.load_namespace("/u/nalln478/ws/exp.multihop_qa/sub.MHQA/config.json")

    in_path = "/u/nalln478/ws/exp.multihop_qa/sub.MHQA/data/dev.json"
    print('Loading test set from {}.'.format(in_path))
    testset, _ = MHQA_data_stream.read_data_file(in_path, FLAGS)
    print('Number of samples: {}'.format(len(testset)))

    right = 0.0
    total = 0.0
    cands_total = 0.0
    for i, (question, passage, entity_start, entity_end, edges, candidates, ref,
            ids, candidates_str) in enumerate(testset):
        if np.argmax(len(x) for x in candidates) == ref:
            right += 1.0
        total += 1.0
        cands_total += len(candidates)
    print('Final performance {}'.format(right/total))
    print('Avg. number of candidates {}'.format(cands_total/total))

