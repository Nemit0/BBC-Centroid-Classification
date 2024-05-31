import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import polars as pl
import cudf as cd

from typing import List
from tqdm import tqdm
from rich import print


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int, max_seq_length: int, pos_dropout: float):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)
        self.fc = nn.Linear(d_model, vocab_size)
        self.init_weights()
        