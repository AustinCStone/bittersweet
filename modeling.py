import torch


import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset


class LearnedPositionEncoding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super(LearnedPositionEncoding, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(max_seq_len, embedding_dim))
        nn.init.uniform_(self.position_embeddings, -0.1, 0.1)

    def forward(self, x):
        # Assuming x is of shape [batch_size, seq_len, embedding_dim]
        seq_len = x.size(1)
        position_embeddings = self.position_embeddings[:seq_len, :]
        position_embeddings = position_embeddings.unsqueeze(axis=0)
        return x + position_embeddings


class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 include_linear: bool = False,
                 vector_input: bool = False,
                 max_len: int = 64):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = LearnedPositionEncoding(max_seq_len=max_len, embedding_dim=d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.vector_input = vector_input
        if not self.vector_input:
            self.embedding = nn.Embedding(ntoken, d_model) # ntoken is 2 and d_model is 128
        self.d_model = d_model
        self.include_linear = include_linear
        if include_linear:
            self.linear = nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        if not self.vector_input:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        if self.include_linear:
            self.linear.bias.data.zero_()
            self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        if not self.vector_input:
            src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        if self.include_linear:
            output = self.linear(output)
        return output
