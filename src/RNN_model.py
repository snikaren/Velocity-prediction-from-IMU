import numpy as np
import pandas as pd
import torch
import torch.nn as nn

"""
---------  Structure --------
Want a many to many RNN. Each input corresponds to output at every timestep


"""


"""
---------- Clarification ----------

seq_len: how many timesteps the model sees contiguously before producing a block of outputs (your memory horizon for training).
batch: how many independent sequences you train on in parallel in one optimizer step. the batch just parallelizes many runs.

Truncated BPTT: 

"""

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
    
    def forward_sequence(self, X, h0 = None):
        # X: (batch, seq_len, input_size)

        h = torch.zeros(X.size(0), self.U.in_features) if h0 is None else h0
        
        outputs = []

        for t in range(X.size(1)):
            y, h = self.forward(X[:,t,:], h)
            outputs.append(y)
        
        Y = torch.stack(outputs, dim = 1)
        
        return Y , h
    

def main():

    features = ["Gyrox","Gyroy","Gyroz"]
    output = ["velocity"]

    #Use to try system
    Feature_data = np.random.rand(100),np.random.rand(100),np.random.rand(100)
    output_data = np.random.rand(100)

    #Define RNN
    model = RNN(RNN(input_size = len(features), hidden_size = 10, output_size = len(output)))

    #Define Training hyperparams
    seq_len = 10
    batch_size = 10
    T = len(output_data)

    X = [batch_size, Feature_data, seq_len]
    Y = [batch_size, output_data, seq_len]

    store_loss = []

    for t in range(0,T,seq_len):

        X_seq = X[: , t:t+seq_len , :]
        Y_seq = Y[: , t:t+seq_len , :]

        Y_pred, h = model.forward_sequence(X_seq, h)

        loss = torch.MSELoss(Y_pred, Y_seq)

        store_loss.append(loss)
    
    print(store_loss)


main()






        



        
