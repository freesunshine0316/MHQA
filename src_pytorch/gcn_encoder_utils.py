
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class GCNEncoder(nn.Module):
    def __init__(self, emb_size, dropout_rate):
        super(GCNEncoder, self).__init__()
        self.weights = nn.Sequential(nn.Linear(emb_size, emb_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate))


    # mention_emb [batch, mention, emb]
    # mention_mask_float [batch, mention]
    # edges [batch, mention, edge]
    # edge_mask_float [batch, mention, edge]
    def forward(self, mention_emb, mention_mask_float, edges, edge_mask_float):
        batch_size, mention_max_size, edge_max_size = list(edges.size())
        edge_emb = utils.batch_gather(mention_emb, edges) * \
                edge_mask_float.unsqueeze(dim=3) # [batch, mention, edge, emb]
        return self.weights(edge_emb.sum(dim=2)) * \
                mention_mask_float.unsqueeze(dim=2) # [batch, mention, emb]

