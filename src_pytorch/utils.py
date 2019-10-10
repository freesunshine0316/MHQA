import os
import errno
import codecs
import collections
import json
import math
import shutil
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_f1(n_both, n_ref, n_out):
    #print('n_out {}, n_ref {}, n_both {}'.format(n_out, n_ref, n_both))
    pr = n_both/n_out if n_out > 0.0 else 0.0
    rc = n_both/n_ref if n_ref > 0.0 else 0.0
    f1 = 2.0*pr*rc/(pr+rc) if pr > 0.0 and rc > 0.0 else 0.0
    return pr, rc, f1


# B I B B I O I I O B O ==> (0,1) (2,2) (3,4) (6,7) (9,9)
def bio_tags_to_spans(tags, tag_len):
    spans = set()
    st = -1
    for i in range(tag_len):
        if tags[i] == 1: # B
            if st != -1:
                assert st <= i-1
                spans.add((st, i-1))
            st = i
        elif tags[i] == 2: # I
            if st == -1:
                st = i
        elif tags[i] == 0: # O
            if st != -1:
                assert st <= i-1
                spans.add((st, i-1))
            st = -1
    return spans


def flatten(l):
    return [item for sublist in l for item in sublist]


def set_gpus(*gpus):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print("Setting CUDA_VISIBLE_DEVICES to: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    return path


def load_char_dict(char_vocab_path):
    vocab = [u"<unk>"]
    with codecs.open(char_vocab_path, encoding="utf-8") as f:
        vocab.extend(l.strip() for l in f.readlines())
    char_dict = collections.defaultdict(int)
    char_dict.update({c:i for i, c in enumerate(vocab)})
    return char_dict


def maybe_divide(x, y):
    return 0 if y == 0 else x / float(y)


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x


def shape(x, dim):
    return list(x.size())[dim]


class FFNN(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size, output_size, dropout, output_weights_initializer=None):
        super(FFNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout) if dropout is not None and dropout > 0.0 else None
        self.linear = []
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.linear.append(nn.Linear(input_size, hidden_size))
            else:
                self.linear.append(nn.Linear(hidden_size, hidden_size))
        last_input_size = hidden_size if self.num_hidden_layers > 0 else input_size
        self.linear.append(nn.Linear(last_input_size, output_size))
        self.linear = nn.ModuleList(self.linear)

    def forward(self, inputs):
        current_inputs = inputs
        for i in range(self.num_hidden_layers):
            current_outputs = F.relu(self.linear[i](current_inputs))
            if self.dropout is not None:
                current_outputs = self.dropout(current_outputs)
            current_inputs = current_outputs
        outputs = self.linear[self.num_hidden_layers](current_inputs)
        return outputs


class CNN1D(nn.Module):
    def __init__(self, input_size, filter_sizes, num_filters):
        super(CNN1D, self).__init__()
        self.input_size = input_size
        self.num_filters = num_filters
        self.convs = { filter_size : nn.Conv1d(input_size, num_filters, filter_size) \
                for filter_size in filter_sizes }

    def forward(inputs):
        outputs = []
        for filter_size, conv_layer in self.convs.items():
            conv = conv_layer(inputs) # [num_words, num_chars - filter_size + 1, num_filters]
            h = F.relu(conv)
            pooled = torch.max(h, dim=1) # [num_words, num_filters]
            outputs.append(pooled)
        return torch.cat(outputs, dim=1) # [num_words, num_filters * len(filter_sizes)]


# emb: [batch, seqlen, emb]
# indices: [batch, x, index] or [batch, index]
def batch_gather(emb, indices):
    batch_size, seq_len = list(emb.size())[:2]
    if len(emb.size()) > 2:
        assert len(emb.size()) == 3
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    flattened_emb = emb.contiguous().view(batch_size * seq_len, emb_size) # [batch_size * seqlen, emb]

    ori_indices = indices
    assert len(indices.size()) > 1 and shape(indices, 0) == batch_size
    if len(indices.size()) == 2:
        x = 1
        num_indices = shape(indices, 1)
        indices = indices.view(batch_size, x, num_indices)
    elif len(indices.size()) == 3:
        x = shape(indices, 1)
        num_indices = shape(indices, 2)
    else:
        assert False

    offset = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, x, num_indices) * seq_len # [batch_size, x, index]
    if torch.cuda.is_available():
        offset = offset.cuda()
    indices = (indices + offset).view(batch_size * x * num_indices) # [batch_size * x * num_indices]
    gathered = torch.index_select(flattened_emb, 0, indices).view(batch_size, x, num_indices, emb_size)
    if len(ori_indices.size()) == 2:
        gathered = gathered.squeeze(dim=1)
    if len(emb.size()) == 2:
        gathered = gathered.squeeze(dim=-1)
    return gathered


def sequence_mask(lens, max_len):
    assert len(lens.size()) == 1
    lens = lens.unsqueeze(dim=1).expand(-1, max_len) # [batch, max_len]
    batch_size = shape(lens, 0)
    indices = torch.arange(max_len).unsqueeze(dim=0).expand(batch_size, -1) # [batch, max_len]
    if torch.cuda.is_available():
        indices = indices.cuda()
    return indices < lens


class RetrievalEvaluator(object):
    def __init__(self):
        self._num_correct = 0
        self._num_gold = 0
        self._num_predicted = 0

    def update(self, gold_set, predicted_set):
        self._num_correct += len(gold_set & predicted_set)
        self._num_gold += len(gold_set)
        self._num_predicted += len(predicted_set)

    def recall(self):
        return maybe_divide(self._num_correct, self._num_gold)

    def precision(self):
        return maybe_divide(self._num_correct, self._num_predicted)

    def metrics(self):
        recall = self.recall()
        precision = self.precision()
        f1 = maybe_divide(2 * recall * precision, precision + recall)
        return recall, precision, f1


class EmbeddingDictionary(object):
    def __init__(self, info, normalize=True):
        self._size = info["emb_dim"]
        self._path = info["path"]
        self._normalize = normalize
        self.load_embedding_dict(self._path)

    @property
    def size(self):
        return self._size

    def load_embedding_dict(self, path):
        print("Loading word embeddings from {}...".format(path))
        default_embedding = np.zeros(self.size)
        self.embedding_dict = collections.defaultdict(lambda:default_embedding)
        with open(path) as f:
            for i, line in enumerate(f.readlines()):
                word_end = line.find(" ")
                word = line[:word_end]
                embedding = np.fromstring(line[word_end + 1:], np.float32, sep=" ")
                if self._normalize:
                    embedding = self.normalize(embedding)
                assert len(embedding) == self.size
                self.embedding_dict[word] = embedding
        print("Done loading word embeddings.")

    def __getitem__(self, key):
        return self._embeddings[key]

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        else:
            return v


if __name__ == '__main__':
    tags = [1, 2, 1, 1, 2, 0, 2, 2, 0, 1, 0]
    print(bio_tags_to_spans(tags, len(tags)))

