import torch


import functools
import math
import numpy as np
import os
from tempfile import TemporaryDirectory
from modeling import expand, average
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from functions import vq, vq_st


@functools.lru_cache
def chunked_triu(sequence_length: int, compression_factor: int):
    """Create a chunked upper triangular mask for the transformer attention mechanism.

    Args:
        sequence_length: int, the length of the sequence
        compression_factor: int, the compression factor. Effects the chunking.
    Returns:
        A chunked attention matrix.
        
        For example, if sequence length is 6 and
        compression factor is 2, the output will be:
        array([[0, 0, 1, 1, 1, 1],
               [0, 0, 1, 1, 1, 1],
               [0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 1, 1],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]])
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert sequence_length % compression_factor == 0
    attention_mat = np.triu(
        np.full((sequence_length, sequence_length), float('-inf'), dtype=np.float32),
        k=1
    )
    chunked_mat = np.repeat(attention_mat, compression_factor, axis=0)
    chunked_mat = np.repeat(chunked_mat, compression_factor, axis=1)
    
    chunked_mat = torch.from_numpy(chunked_mat).to(device)
    return chunked_mat


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


class PoolExpandTransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers_pre: int, nlayers_post: int, dropout: float = 0.5,
                 include_linear: bool = False,
                 vector_input: bool = False,
                 max_len: int = 64,
                 compression_factor=1):
        super().__init__()
        self.compression_factor = compression_factor
        if self.compression_factor < 1:
            self.expand_linear = nn.Linear(int(d_model * compression_factor), d_model)
        self.max_len = max_len
        self.model_type = 'Transformer'
        self.pos_encoder = LearnedPositionEncoding(max_seq_len=max_len, embedding_dim=d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.pre_transformer_encoder = TransformerEncoder(encoder_layers, nlayers_pre)
        self.post_transformer_encoder = TransformerEncoder(encoder_layers, nlayers_post)
        self.vector_input = vector_input
        if not self.vector_input:
            self.embedding = nn.Embedding(ntoken, d_model) # ntoken is 2 and d_model is 128
        self.d_model = d_model
        self.include_linear = include_linear
        if include_linear:
            self.linear = nn.Linear(d_model, ntoken)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_encoder.to(device)
        self.pre_transformer_encoder.to(device)
        self.post_transformer_encoder.to(device)
        if not self.vector_input:
            self.embedding.to(device)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        if not self.vector_input:
            self.embedding.weight.data.uniform_(-initrange, initrange)
        if self.include_linear:
            self.linear.bias.data.zero_()
            self.linear.weight.data.uniform_(-initrange, initrange)
        if self.compression_factor < 1:
            self.expand_linear.bias.data.zero_()
            self.expand_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_len, seq_len]`` if not vector_input
                else ``[batch_len, seq_len, d_model]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]`` if include_linear
                else ``[batch_size, seq_len, d_model]``
        """
        if not self.vector_input:
            src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if self.compression_factor < 1:
            sequence_length = int(self.max_len * self.compression_factor)
            compression_factor = 1
        else:
            sequence_length = self.max_len
            compression_factor = self.compression_factor
        chunked_mask = chunked_triu(sequence_length=sequence_length,
                                    compression_factor=compression_factor)
        output = self.pre_transformer_encoder(src, is_causal=True,
                                              mask=chunked_mask)
        if self.compression_factor > 1:
            output = average(output, self.compression_factor)
            sequence_length = int(self.max_len / self.compression_factor)
            compression_factor = 1
        elif self.compression_factor < 1:
            output = expand(output, self.compression_factor)
            sequence_length = self.max_len
            compression_factor = int(1 / self.compression_factor)
            output = self.expand_linear(output)
        elif self.compression_factor == 1:
            sequence_length = self.max_len
            compression_factor = 1

        chunked_mask = chunked_triu(sequence_length=sequence_length,
                                    compression_factor=compression_factor)
        output = self.post_transformer_encoder(output, is_causal=True,
                                               mask=chunked_mask)
        if self.include_linear:
            return self.linear(output)
        return output

