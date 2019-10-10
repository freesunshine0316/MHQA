
import numpy as np

import grn_encoder_utils
import gcn_encoder_utils
import utils

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = \
        "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = \
        "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


class ModelGraph(nn.Module):
    def __init__(self, general_config):
        super(ModelGraph, self).__init__()

        self.general_config = general_config
        self.dropout = nn.Dropout(self.general_config.dropout)

        self.elmo_L = 2
        self.elmo = Elmo(options_file, weight_file, self.elmo_L, dropout=self.general_config.dropout)
        self.elmo_weights = nn.Linear(self.elmo_L, 1, bias=False)

        emb_size = 1024
        if self.general_config.embedding_compress_size > 0:
            self.embedding_projector = nn.Linear(emb_size, self.general_config.embedding_compress_size)
            emb_size = self.general_config.embedding_compress_size

        if self.general_config.use_mention_feature:
            self.mention_width_embedding = nn.Embedding(self.general_config.mention_max_width,
                    self.general_config.feature_size)
            self.mention_type_embedding = nn.Embedding(4, self.general_config.feature_size)

        mention_emb_size = emb_size * 2
        if self.general_config.use_mention_head:
            mention_emb_size += emb_size
        if self.general_config.use_mention_feature:
            mention_emb_size += 2 * self.general_config.feature_size

        self.mention_scorer = utils.FFNN(self.general_config.ffnn_depth,
                mention_emb_size, self.general_config.ffnn_size, 1, self.general_config.dropout)

        if self.general_config.use_mention_head:
            self.mention_head_scorer = nn.Linear(emb_size, 1)

        if self.general_config.graph_encoding == "GRN":
            print("With GRN as the graph encoder")
            self.graph_encoder = grn_encoder_utils.GRNEncoder(mention_emb_size, self.general_config.dropout)
        elif self.general_config.graph_encoding == "GCN":
            print("With GCN as the graph encoder")
            self.graph_encoder = gcn_encoder_utils.GCNEncoder(mention_emb_size, self.general_config.dropout)
        else:
            self.graph_encoder = None

        if self.general_config.matching_op == "concat":
            self.concat_projector = nn.Linear(mention_emb_size * 2, 1)

        if self.general_config.graph_encoding in ("GRN", "GCN"):
            self.matching_integrater = nn.Linear(general_config.graph_encoding_steps + 1, 1, bias=False)


    def get_elmo_repre(self, ids):
        elmo_outputs = self.elmo(ids)
        repre = torch.stack(elmo_outputs['elmo_representations'], dim=3) # [batch, seq, emb, L]
        repre = self.elmo_weights(repre).squeeze(dim=3) # [batch, seq, emb]
        return repre * elmo_outputs['mask'].float().unsqueeze(dim=2) # [batch, seq, emb]


    def forward(self, batch):
        passage_repre = self.get_elmo_repre(batch['passage_ids']) # [batch, passage, emb]
        question_repre = self.get_elmo_repre(batch['question_ids']) # [batch, question, emb]

        # modeling question
        batch_size = len(batch['ids'])
        question_starts = torch.zeros(batch_size, 1, dtype=torch.long).cuda() # [batch, 1]
        question_ends = batch['question_lens'].view(batch_size, 1) - 1 # [batch, 1]
        question_types = torch.zeros(batch_size, 1,  dtype=torch.long).cuda() # [batch, 1]
        question_mask_float = torch.ones(batch_size, 1, dtype=torch.float).cuda() # [batch, 1]
        question_emb = self.get_mention_embedding(question_repre, question_starts, question_ends,
                question_types, question_mask_float).squeeze(dim=1) # [batch, emb]

        # modeling mentions
        mention_starts = batch['mention_starts']
        mention_ends = batch['mention_ends']
        mention_types = batch['mention_types']
        mention_nums = batch['mention_nums']

        mention_max_num = utils.shape(mention_starts, 1)
        mention_mask = utils.sequence_mask(mention_nums, mention_max_num)
        mention_emb = self.get_mention_embedding(passage_repre, mention_starts, mention_ends,
                mention_types, mention_mask.float())

        if self.general_config.embedding_compress_size > 0:
            question_emb = self.embedding_projector(question_emb)
            mention_emb = self.embedding_projector(mention_emb)

        matching_results = []
        rst_seq = self.perform_matching(mention_emb, question_emb)
        matching_results.append(rst_seq)

        # graph encoding
        if self.general_config.graph_encoding in ('GCN', 'GRN'):
            if self.general_config.graph_encoding in ("GRN", "GCN"):
                edges = batch['edges'] # [batch, mention, edge]
                edge_nums = batch['edge_nums'] # [batch, mention]
                edge_max_num = utils.shape(edges, 2)
                edge_mask = utils.sequence_mask(edge_nums.view(batch_size * mention_max_num),
                        edge_max_num).view(batch_size, mention_max_num, edge_max_num) # [batch, mention, edge]
                #assert not (edge_mask & (~mention_mask.unsqueeze(dim=2))).any().item()

            for i in range(self.general_config.graph_encoding_steps):
                # TODO: consider merging question emb into mentions to make 'question-aware'
                mention_emb_new = self.graph_encoder(mention_emb, mention_mask.float(), edges, edge_mask.float())
                mention_emb = mention_emb_new + mention_emb if self.general_config.graph_residual else mention_emb_new
                rst_graph = self.perform_matching(mention_emb, question_emb)
                matching_results.append(rst_graph)

        if len(matching_results) > 1:
            assert len(matching_results) == self.general_config.graph_encoding_steps + 1
            matching_results = torch.stack(matching_results, dim=2) # [batch, mention, graph_step+1]
            logits = self.matching_integrater(matching_results).squeeze(dim=2) # [batch, mention]
        else:
            assert len(matching_results) == 1
            logits = matching_results[0] # [batch, mention]

        candidates, candidate_num, candidate_appear_num = \
                batch['candidates'], batch['candidate_num'], batch['candidate_appear_num']
        _, cand_max_num, cand_pos_max_num = list(candidates.size())

        candidate_mask = utils.sequence_mask(candidate_num, cand_max_num) # [batch, cand]
        candidate_appear_mask = utils.sequence_mask(candidate_appear_num.view(batch_size * cand_max_num),
                cand_pos_max_num).view(batch_size, cand_max_num, cand_pos_max_num) # [batch, cand, pos]
        #assert not (candidate_appear_mask & (~candidate_mask.unsqueeze(dim=2))).any().item()

        candidate_appear_logits = utils.batch_gather(logits, candidates) + \
                candidate_appear_mask.float().log() # [batch, cand, pos]
        candidate_logits = candidate_appear_logits.max(dim=2)[0] # [batch, cand]

        predictions = candidate_logits.argmax(dim=1) # [batch]
        assert (predictions < candidate_num).all()

        if 'refs' not in batch or batch['refs'] is None:
            return {'predictions': predictions}

        refs = batch['refs']
        loss = nn.CrossEntropyLoss()(candidate_logits, refs)
        right_count = (predictions == refs).sum()
        return {'predictions':predictions, 'loss':loss, 'right_count':right_count}


    # input_repre: [batch, seq, emb]
    # mention_starts, mention_ends and mention_mask: [batch, mentions]
    # s_m(i) = FFNN(g_i)
    # g_i = [x_i^start, x_i^end, x_i^head, \phi(i)]
    def get_mention_embedding(self, input_repre, mention_starts, mention_ends, mention_types, mention_mask_float):
        mention_emb_list = []
        mention_start_emb = utils.batch_gather(input_repre, mention_starts) # [batch, mentions, emb]
        mention_emb_list.append(mention_start_emb)
        mention_end_emb = utils.batch_gather(input_repre, mention_ends) # [batch, mentions, emb]
        mention_emb_list.append(mention_end_emb)

        if self.general_config.use_mention_head:
            batch_size, mention_num = list(mention_starts.size())

            span_starts = mention_starts.unsqueeze(dim=2) # [batch, mentions, 1]
            span_ends = mention_ends.unsqueeze(dim=2) # [batch, mentions, 1]
            span_range = torch.arange(self.general_config.mention_max_width).view(1, 1,
                    self.general_config.mention_max_width) # [1, 1, span_width]
            if torch.cuda.is_available():
                span_range = span_range.cuda()
            span_indices_raw = span_starts + span_range # [batch, mentions, span_width]
            span_indices = torch.min(span_indices_raw, span_ends) # [batch, mentions, span_width]
            span_mask = span_indices_raw <= span_ends # [batch, mention, span_width]

            span_emb = utils.batch_gather(input_repre, span_indices) # [batch, mentions, span_width, emb]
            span_scores = self.mention_head_scorer(span_emb).squeeze(dim=-1) + \
                    torch.log(span_mask.float()) # [batch, mentions, seq_width]
            span_attn = F.softmax(span_scores, dim=-1) # [batch, mentions, span_width]
            mention_head_emb = span_emb * span_attn.unsqueeze(dim=-1) # [batch, mentions, span_width, emb]
            mention_head_emb = torch.sum(mention_head_emb, dim=2) # [batch, mentions, emb]
            mention_emb_list.append(mention_head_emb)

        if self.general_config.use_mention_feature:
            mention_width = 1 + mention_ends - mention_starts # [batch, mentions]
            mention_width_index = torch.clamp(mention_width, 1, self.general_config.mention_max_width) - 1 # [batch, mentions]
            mention_width_emb = self.mention_width_embedding(mention_width_index) # [batch, mentions, emb]
            mention_width_emb = self.dropout(mention_width_emb)
            mention_emb_list.append(mention_width_emb)
            mention_type_emb = self.mention_type_embedding(mention_types)
            mention_emb_list.append(mention_type_emb)

        return torch.cat(mention_emb_list, dim=2) * mention_mask_float.unsqueeze(dim=2)


    # mention_emb: [batch, mention, emb]
    # question_emb: [batch, emb]
    def perform_matching(self, mention_emb, question_emb):
        if self.general_config.matching_op == "matmul":
            # [batch, mention, emb] * [batch, emb, 1] ==> [batch, mention, 1]
            logits = mention_emb.matmul(question_emb.unsqueeze(dim=2)).squeeze(dim=2)
            return logits
        elif self.general_config.matching_op == "concat":
            mention_max_num = utils.shape(mention_emb, 1)
            question_emb = question_emb.unsqueeze(dim=1).expand(-1, mention_max_num, -1)
            combined_emb = torch.cat([mention_emb, question_emb], dim=2)
            logits = self.concat_projector(combined_emb).squeeze(dim=2)
            return logits
        else:
            assert False, "Unsupported matching_op: {}".format(self.general_config.matching_op)

