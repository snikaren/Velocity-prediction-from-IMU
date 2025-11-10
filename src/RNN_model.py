import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):

        self.input_size = input_size #The number of expected features in the input x
        self.hidden_size = hidden_size #The number of features in the hidden state h
        self.num_layers = num_layers #Number of recurrent layers. 

        rnn = nn.RNN(input_size, hidden_size, num_layers)
