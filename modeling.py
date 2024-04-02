import torch


import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from functions import vq, vq_st


class VQEmbedding(nn.Module):

    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # Assumed that z_e_x is of shape (batch size, sequence length, embedding size)
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        # Assumed that z_e_x is of shape (batch size, sequence length, embedding size)
        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()
        return z_q_x, z_q_x_bar


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

def split(x, compression_factor):
    tokens_to_take = int(x.shape[1] / compression_factor)
    sub_x = x[:, :tokens_to_take, :]
    assert sub_x.shape[-1] % 2 == 0
    half1, half2 = sub_x.split(int(sub_x.shape[-1] / 2), dim=2)
    half1_reshaped = half1.unsqueeze(2) 
    half2_reshaped = half2.unsqueeze(2)
    interleaved = torch.cat((half1_reshaped, half2_reshaped), dim=2)
    interleaved = interleaved.view(x.shape[0], x.shape[1], int(x.shape[-1] / 2))
    return interleaved

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5,
                 include_linear: bool = False,
                 vector_input: bool = False,
                 max_len: int = 64,
                 num_latent_vectors=512,
                 use_vq=False,
                 compression_factor=None):
        super().__init__()
        self.use_vq = use_vq
        self.compression_factor = compression_factor
        if self.use_vq:
            self.codebook = VQEmbedding(num_latent_vectors, d_model)
        self.model_type = 'Transformer'
        self.pos_encoder = LearnedPositionEncoding(max_seq_len=max_len, embedding_dim=d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.vector_input = vector_input
        if not self.vector_input:
            self.embedding = nn.Embedding(ntoken, d_model) # ntoken is 2 and d_model is 128
        self.d_model = d_model
        self.include_linear = include_linear
        if include_linear:
            self.linear = nn.Linear(d_model, ntoken)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_encoder.to(device)
        self.transformer_encoder.to(device)
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
        output = self.transformer_encoder(src, is_causal=False)
        if self.include_linear:
            output = self.linear(output)
        if self.use_vq:
            hard_output_st, hard_output = self.codebook.straight_through(output)
            if self.compression_factor is not None:
                assert self.compression_factor == 2
                # Split and concat...
                hard_output_st = split(hard_output_st, self.compression_factor)
                hard_output = split(hard_output, self.compression_factor)
                output = split(output, self.compression_factor)
            return hard_output_st, hard_output, output # Return hard predictions and soft predictions
        if self.compression_factor is not None:
            assert self.compression_factor == 2
            output = split(output, self.compression_factor)
        return output

