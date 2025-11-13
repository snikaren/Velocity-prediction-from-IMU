import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
"""
---------  Structure --------
Want a many to many RNN. Each input corresponds to output at every timestep
"""

"""
---------- Clarification ----------

seq_len: how many timesteps the model sees contiguously before producing a block of outputs (your memory horizon for training).
batch: how many independent sequences you train on in parallel in one optimizer step. the batch just parallelizes many runs.


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
    
    """This function takes a batch and calculates the outputs every step along each sequence
    Batch
    [
    seq1 = [[x7_0],[x7_1],[x7_2],[x7_3],[x7_4]],
    seq2 = [[x1_0],[x1_1],[x1_2],[x1_3],[x1_4]],
    seq3 = [[x3_0],[x3_1],[x3_2],[x3_3],[x3_4]],
    ]
    
    """
    def forward_sequence(self, X):
        # X: (batch, seq_len, input_size)

        h = torch.zeros(X.size(0), self.U.in_features)
        
        outputs = []
    
        for t in range(X.size(1)):
            y, h = self.forward(X[:,t,:], h)
            outputs.append(y)
        
        Y = torch.stack(outputs, dim = 1)

        
        return Y
    

def main():

    features = ["Gyrox"]
    output = ["velocity"]

    """Random data for test"""
    feature_data = torch.randn(100,1)
    output_data = torch.randn(100,1)

    epochs = 10
    batch_size =  96
    seq_len = 5
    T = len(feature_data)
    num_windows = T-seq_len+1

    """Define RNN, Loss function and optimizer"""
    model = RNN(input_size = len(features), hidden_size = 10, output_size = 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    """Create sliding windows"""
    X_windows = torch.stack([feature_data[i:i+seq_len] for i in range(num_windows)], dim=0)   #X_windows.shape = (num_windows, seq_len, 1)
    Y_windows = torch.stack([output_data[i:i+seq_len] for i in range(num_windows)], dim = 0)

    dataset = TensorDataset(X_windows, Y_windows) # dataset[i] = (X_windows[i], Y_windows[i])

    """Training"""
    for epoch in range(epochs):
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #TODO: Look at droplast()? loader is an iterable of batches

        for batch_idx, (X_batch,Y_batch) in enumerate(loader):

            #X_batch: (batch_size, seq_len, 1)
            #before computing new gradients for the current batch, you must clear the old ones:
            optimizer.zero_grad()
            """
            [
            [[x7_0],[x7_1],[x7_2],[x7_3],[x7_4]],
            [[x1_0],[x1_1],[x1_2],[x1_3],[x1_4]],
            [[x3_0],[x3_1],[x3_2],[x3_3],[x3_4]],
            ]
            """
            Y_pred = model.forward_sequence(X_batch)

            loss = criterion(Y_pred, Y_batch)

            loss.backward() #BBTT
            optimizer.step()

            print(loss)

main()


"""
model predicts:
Y_pred with shape (batch, seq_len, 1)
labels are:
Y_batch with shape (batch, seq_len, 1)

nn.MSELoss(): 
-> the loss includes all timesteps in all sequences of the batch.

"""






        



        
