import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
"""
---------  Structure --------
Want a many to many RNN. Each input corresponds to output at every timestep
"""

"""
---------- Clarification ----------

seq_len: how many timesteps the model sees contiguously before producing a block of outputs (your memory horizon for training).
batch: how many independent sequences you train on in parallel in one optimizer step. the batch just parallelizes many runs.


"""

#----- Regular RNN -----
class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):

        super(RNN,self).__init__()

        self.input_size = input_size #The number of features
        self.hidden_size = hidden_size #The number of features in the hidden state h
        self.output_size = output_size

        self.W = nn.Linear(input_size, hidden_size, bias = False)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, output_size)

    def forward(self, x_t, h_prev):

        h_t = torch.tanh(self.W(x_t) + self.U(h_prev))

        y_t = self.V(h_t)

        return y_t, h_t
    
    """This function takes a batch and calculates the outputs every step along each sequence
    Batch
    [
    seq1 = [[x7_0],[x7_1],[x7_2],[x7_3],[x7_4]],
    seq2 = [[x1_0],[x1_1],[x1_2],[x1_3],[x1_4]],
    seq3 = [[x3_0],[x3_1],[x3_2],[x3_3],[x3_4]],
    ]
    """
    def forward_sequence(self, X, pred_len):
        # X: (batch, seq_len, input_size)

        seq_len = X.size(1)
        batch_size = X.size(0)

        h = torch.zeros(X.size(0), self.U.in_features)
        
        predicted_outputs = []
    
        for t in range(seq_len):
            _, h = self.forward(X[:,t,:], h)


        x_future = torch.zeros(batch_size, self.input_size)  # device=X.device GPU thing.

        for _ in range(pred_len):
            y, h = self.forward(x_future, h)
            predicted_outputs.append(y)
        
        Y = torch.stack(predicted_outputs, dim = 1)
        
        return Y


#----- GRU RNN -----
class GRURNN(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True) # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature)

        #Output
        self.V = nn.Linear(hidden_size, output_size)
    
    def forward_sequence(self, X, pred_len):
        # X: (batch, seq_len, input_size)
        batch_size = X.size(0)

        """
        _: (batch, seq_len, hidden_size)
        h: (num_layers * num_directions, batch, hidden_size)
        """
        _ , h = self.rnn(X) #Encode past seq into final hidden state

        #GRU wants (batch, seq_len = 1, input_size) seq_len = how many time steps you feed
        x_future = torch.zeros(batch_size, 1, self.input_size)  # device=X.device GPU thing.

        predicted_outputs = []
        for _ in range(pred_len):
            y, h = self.rnn(x_future,h)

            y = self.V(y.squeeze(1)) # (batch, output_size)
            predicted_outputs.append(y)
        
        Y = torch.stack(predicted_outputs, dim = 1)

        return Y 







        



        
