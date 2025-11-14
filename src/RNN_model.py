import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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


    """Structure Data"""
    IMU = pd.read_csv("Data/IMU_data/data.csv")
    groundtruth = pd.read_csv("Data/state_groundtruth_data/data.csv")

    IMU.drop(index = 0, inplace = True) #Because of different timestamps

    #Adjust such that timestamp start at 0 and so that time-spaces are even in Input and output

    IMU['t'] = (IMU['#timestamp [ns]'] - IMU['#timestamp [ns]'].iloc[0]) * 1e-9
    groundtruth['t'] = (groundtruth['#timestamp'] - groundtruth['#timestamp'].iloc[0]) * 1e-9
    IMU = IMU.iloc[:len(groundtruth)]

    #Write to tensors
    IMU.drop(columns=['#timestamp [ns]','t'], inplace=True)
    X_np = IMU.to_numpy()

    groundtruth_x_vel = groundtruth[[' v_RS_R_x [m s^-1]']]
    Y_np = groundtruth_x_vel.to_numpy()

    X = torch.from_numpy(X_np).float()
    Y = torch.from_numpy(Y_np).float()


    """Random data for test"""

    epochs = 1
    batch_size =  1000
    seq_len = 10

    T = len(X)
    num_windows = T-seq_len+1
    nr_of_features = X.shape[1]
    nr_of_hidden_neurons = 10
    nr_of_outputs = Y.shape[1]

    """Define RNN, Loss function and optimizer"""
    model = RNN(input_size = nr_of_features, hidden_size = nr_of_hidden_neurons, output_size = nr_of_outputs)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    """Create sliding windows"""
    X_windows = torch.stack([X[i:i+seq_len] for i in range(num_windows)], dim = 0)   #X_windows.shape = (num_windows, seq_len, features)
    Y_windows = torch.stack([Y[i:i+seq_len] for i in range(num_windows)], dim = 0)


    dataset = TensorDataset(X_windows, Y_windows) # dataset[i] = (X_windows[i], Y_windows[i])

    """Training"""
    store_mse_loss = []
    for epoch in range(epochs):
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) #TODO: Look at droplast()? loader is an iterable of batches

        for batch_idx, (X_batch,Y_batch) in enumerate(loader):

            #X_batch: (batch_size, seq_len, 1)
            #before computing new gradients for the current batch, you must clear the old ones:
            optimizer.zero_grad()
            
            #[
            #[[x7_0],[x7_1],[x7_2],[x7_3],[x7_4]],
            #[[x1_0],[x1_1],[x1_2],[x1_3],[x1_4]],
            #[[x3_0],[x3_1],[x3_2],[x3_3],[x3_4]],
            #]
           
            Y_pred = model.forward_sequence(X_batch)

            loss = criterion(Y_pred, Y_batch) #Many-many loss over the entire batch

            loss.backward() # BPTT computes ∂L/∂W, ∂L/∂U, ∂L/∂V, etc
            optimizer.step() # one call to optimizer.step() = one weight update using the current batch’s gradients.

            store_mse_loss.append(loss.item())
    
    plt.plot(store_mse_loss)
    plt.xlabel("Batch")
    plt.ylabel("MSE")
    plt.title("MSE loss over batches")
    plt.show()
        
main()


"""
model predicts:
Y_pred with shape (batch, seq_len, 1)
labels are:
Y_batch with shape (batch, seq_len, 1)

nn.MSELoss(): 
-> the loss includes all timesteps in all sequences of the batch.

"""






        



        
