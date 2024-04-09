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



def expand(x: torch.Tensor, compression_factor: float):
    """Expand a tensor along the sequence dimension by interleaving adjacent elements along the feature dimension.
    
    Args:
        x: torch.Tensor, shape [batch_size, seq_len, d_model]
        compression_factor: float, the 1 / factor by which to expand the tensor along the sequence dimension.
    
    Returns:
        interleaved: torch.Tensor, shape [batch_size, int(seq_len / compression_factor), int(d_model * compression_factor)]
    """
    # Convert compression_factor to expansion factor (integer)
    expansion_factor = int(1 / compression_factor)
    assert compression_factor < 1 and expansion_factor > 1, "compression_factor must be less than 1 and its inverse should be an integer"
    
    # Split the tensor into `expansion_factor` parts along the last dimension
    split_tensors = x.split(x.shape[-1] // expansion_factor, dim=2)
    
    # Initialize a list to store reshaped splits
    reshaped_splits = []
    
    for split in split_tensors:
        # Unsqueeze to make room for interleaving
        reshaped_splits.append(split.unsqueeze(2))
    
    # Concatenate along the new axis to interleave, and then reshape to merge interleaved and sequence dimensions
    interleaved = torch.cat(reshaped_splits, dim=2)
    interleaved = interleaved.view(x.shape[0], x.shape[1] * expansion_factor, -1)
    
    return interleaved




def average(x: torch.Tensor, compression_factor: int) -> torch.Tensor:
    """Average `compression_factor` adjacent elements in a tensor along the feature dimension.

    Args:
    x: torch.Tensor, shape [batch_size, seq_len, d_model]
    compression_factor: int, the factor by which to compress the tensor along the feature dimension.

    Returns:
    averaged_x: torch.Tensor, shape [batch_size, seq_len // compression_factor, d_model]
    """
    # Check if seq_len is divisible by compression_factor
    assert x.shape[1] % compression_factor == 0, "seq_len must be divisible by compression_factor"
    
    # Reshape to group elements based on the compression factor
    # New shape: [batch_size, seq_len // compression_factor, compression_factor, d_model]
    x = x.view(x.size(0), x.shape[1] // compression_factor, compression_factor, x.size(2))
    
    # Compute the mean along the new dimension to average grouped elements
    averaged_x = x.mean(dim=2)
    return averaged_x


class PoolExpandTransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers_pre: int, nlayers_post: int, dropout: float = 0.5,
                 include_linear: bool = False,
                 vector_input: bool = False,
                 max_len: int = 64,
                 num_latent_vectors=512,
                 use_vq=False,
                 compression_factor=1):
        super().__init__()
        self.use_vq = use_vq
        self.compression_factor = compression_factor
        self.codebook = VQEmbedding(num_latent_vectors, d_model)
        if self.compression_factor < 1:
            self.expand_linear = nn.Linear(int(d_model * compression_factor), d_model)
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

    def set_codebook(self, codebook_weights):
        self.codebook.embedding.weight.data = codebook_weights

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
        output = self.pre_transformer_encoder(src, is_causal=False)
        if self.compression_factor > 1:
            output = average(output, self.compression_factor)
        elif self.compression_factor < 1:
            output = expand(output, self.compression_factor)
            output = self.expand_linear(output)
        output = self.post_transformer_encoder(output, is_causal=False)
        if self.include_linear:
            output = self.linear(output)
        if self.use_vq:
            tokens = self.codebook(output)
            hard_output_st, hard_output = self.codebook.straight_through(output)
            return hard_output_st, hard_output, output, tokens
        else:
            return output

