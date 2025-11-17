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
        
    

def main():

    """Structure Data"""
    #TODO Normalize data ? 
    IMU = pd.read_csv(r"C:\Users\hampu\Desktop\RNN_test\Velocity-prediction-from-IMU\Data\IMU_data\data.csv")
    groundtruth = pd.read_csv(r"C:\Users\hampu\Desktop\RNN_test\Velocity-prediction-from-IMU\Data\state_groundtruth_data\data.csv")

    IMU.drop(index = 0, inplace = True) #Because of different timestamps

    #Adjust such that timestamp start at 0 and so that time-spaces are even in Input and output

    IMU['t'] = (IMU['#timestamp [ns]'] - IMU['#timestamp [ns]'].iloc[0]) * 1e-9
    groundtruth['t'] = (groundtruth['#timestamp'] - groundtruth['#timestamp'].iloc[0]) * 1e-9
    IMU = IMU.iloc[:len(groundtruth)] #Ensure same length

    #Write to tensors
    IMU.drop(columns=['#timestamp [ns]','t'], inplace=True)
    X_np = IMU.to_numpy()

    groundtruth_x_vel = groundtruth[[' v_RS_R_x [m s^-1]']]
    Y_np = groundtruth_x_vel.to_numpy()

    X = torch.from_numpy(X_np).float() 
    Y = torch.from_numpy(Y_np).float()

    epochs = 100
    batch_size =  150
    seq_len = 20
    pred_len = 5
    test_ratio = 0.2
    val_ratio = 0.2


    nr_of_features = X.shape[1]
    nr_of_hidden_neurons = 5
    nr_of_outputs = Y.shape[1]

    """Define RNN, Loss function and optimizer"""
    model = RNN(input_size = nr_of_features, hidden_size = nr_of_hidden_neurons, output_size = nr_of_outputs)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    """Create sliding windows"""
    tot_timesteps = len(X)
    num_windows = tot_timesteps-seq_len-pred_len+1 

    X_windows = torch.stack([X[i:i+seq_len] for i in range(num_windows)], dim = 0)   #X_windows.shape = (num_windows, seq_len, features)
    Y_windows = torch.stack([Y[i+seq_len:i+seq_len+pred_len] for i in range(num_windows)], dim = 0)

    """Create dataset from windows"""
    dataset = TensorDataset(X_windows, Y_windows) # dataset[i] = (X_windows[i], Y_windows[i])
    data_size = len(dataset) # nr of windows

    """Create train and HOLDOUT test dataset"""
    n_test = int(test_ratio * data_size)
    n_train = data_size-n_test

    train, test = random_split(dataset,[n_train,n_test]) #Do not touch the test set

    """-----Training-----"""

    """Create test and validation dataset"""
    n_validation = int(val_ratio * len(train))
    n_train_dataset = len(train)-n_validation

    train_dataset, validation_dataset = random_split(train,[n_train_dataset,n_validation])

    training_loss = []
    validation_loss = []

    print(len(validation_dataset))


    for epoch in range(epochs):

        print(f"----STARTING EPOCH: {epoch} ----")

        model.train() #training mode
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #TODO: Look at droplast()? loader is an iterable of batches
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

        epoch_train_loss = 0
        epoch_validation_loss = 0

        #------TRAINING-----
        for X_batch,Y_batch in train_loader:

            #X_batch: (batch_size, seq_len, 1)
            #before computing new gradients for the current batch, you must clear the old ones:
            optimizer.zero_grad()
            
            #e.g)
            #[
            #[[x7_0],[x7_1],[x7_2],[x7_3],[x7_4]],
            #[[x1_0],[x1_1],[x1_2],[x1_3],[x1_4]],
            #[[x3_0],[x3_1],[x3_2],[x3_3],[x3_4]],
            #]

           #Get y_pred for whole batch
            Y_pred_batch = model.forward_sequence(X_batch,pred_len)

            loss = criterion(Y_pred_batch, Y_batch) #Many-many loss over the entire batch

            loss.backward() # BPTT computes ∂L/∂W, ∂L/∂U, ∂L/∂V, etc
            optimizer.step() # one call to optimizer.step() = one weight update using the current batch’s gradients

            epoch_train_loss += loss.item()

        #------Validation-----
        model.eval()
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:

                Y_pred = model.forward_sequence(X_batch,pred_len)
                loss = criterion(Y_pred, Y_batch)
                epoch_validation_loss += loss.item()
        
        training_loss.append(epoch_train_loss / len(train_loader))
        validation_loss.append(epoch_validation_loss / len(val_loader))

    plt.figure()
    plt.plot(training_loss, label = 'Training loss', color = 'blue')
    plt.plot(validation_loss, label = 'validation loss', color = 'red')
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE loss')
    plt.legend()
    plt.grid(True)

    plt.show()
    

main()








        



        
