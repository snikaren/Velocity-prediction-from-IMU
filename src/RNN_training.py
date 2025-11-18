import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

"""From other files"""
from RNN_model import RNN,GRURNN
from config import *


def data_handling(IMU,groundtruth):

    IMU.drop(index = 0, inplace = True) #Because of different timestamps

    #Adjust such that timestamp start at 0 and so that time-spaces are even in Input and output

    IMU['t'] = (IMU['#timestamp [ns]'] - IMU['#timestamp [ns]'].iloc[0]) * 1e-9
    groundtruth['t'] = (groundtruth['#timestamp'] - groundtruth['#timestamp'].iloc[0]) * 1e-9
    IMU = IMU.iloc[:len(groundtruth)] #Ensure same length

    IMU.drop(columns=['#timestamp [ns]','t'], inplace=True)

    #Split into train and test set
    tot_timesteps = IMU.shape[0]
    train_size = int((1-test_ratio) * tot_timesteps)

    train_IMU = IMU[0:train_size]
    test_IMU = IMU[train_size:]

    train_groundtruth = groundtruth[0:train_size]
    test_groundtruth = groundtruth[train_size:]

    #Write to tensors
    X_np_train = train_IMU.to_numpy()
    X_np_test = test_IMU.to_numpy()

    X_train = torch.from_numpy(X_np_train).float() 
    X_test = torch.from_numpy(X_np_test).float()

    groundtruth_x_vel_train = train_groundtruth[[' v_RS_R_x [m s^-1]']]
    groundtruth_x_vel_test = test_groundtruth[[' v_RS_R_x [m s^-1]']]

    Y_np_train = groundtruth_x_vel_train.to_numpy()
    Y_np_test = groundtruth_x_vel_test.to_numpy()

    Y_train = torch.from_numpy(Y_np_train).float()
    Y_test = torch.from_numpy(Y_np_test).float()

    nr_of_features = X_train.shape[1]
    nr_of_outputs = Y_train.shape[1]

    return X_train, Y_train, X_test, Y_test, nr_of_features, nr_of_outputs


def Create_sliding_windows(X,Y, seq_len, pred_len):
    tot_timesteps = len(X)
    num_windows = tot_timesteps-seq_len-pred_len+1 

    X_windows = torch.stack([X[i:i+seq_len] for i in range(num_windows)], dim = 0)   #X_windows.shape = (num_windows, seq_len, features)
    Y_windows = torch.stack([Y[i+seq_len:i+seq_len+pred_len] for i in range(num_windows)], dim = 0)

    return X_windows, Y_windows


def RNN_training(X_windows,Y_windows,nr_of_features,nr_of_outputs,model_choice,epochs,batch_size,seq_len,pred_len,learning_rate,nr_of_hidden_neurons):

    """Define RNN, Loss function and optimizer"""
    if model_choice == "RNN":
        model = RNN(input_size = nr_of_features, hidden_size = nr_of_hidden_neurons, output_size = nr_of_outputs)
    
    elif model_choice == "GRU":
        model = GRURNN(input_size = nr_of_features, hidden_size = nr_of_hidden_neurons, output_size = nr_of_outputs)
    else:
        raise Exception("No RNN model defined")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


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


    for i in range(epochs):

        print(f"----STARTING EPOCH: {epochs} ----")

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


def RNN_main_pipeline():

    """Structure Data"""
    #TODO Normalize data ? 
    IMU = pd.read_csv(r"C:\Users\hampu\Desktop\RNN_test\Velocity-prediction-from-IMU\Data\IMU_data\data.csv")
    groundtruth = pd.read_csv(r"C:\Users\hampu\Desktop\RNN_test\Velocity-prediction-from-IMU\Data\state_groundtruth_data\data.csv")

    X, Y, nr_of_features, nr_of_outputs = data_handling(IMU,groundtruth)
    dataset = TensorDataset(X, Y) # dataset[i] = (X[i], Y[i])

    train, test = random_split(dataset,[n_train,n_test]) #Never touch test set in training


    """Set up training for various cases"""
    for seq_len in seq_len_list:
        for pred_len in pred_len_list:
            X_windows, Y_windows = Create_sliding_windows(X,Y, seq_len, pred_len)

            for model_choice in model_choice_list:
                for epochs in epochs_list:
                    for batch_size in batch_size_list:
                                for learning_rate in learning_rate_list:
                                    for nr_of_hidden_neurons in nr_of_hidden_neurons_list:
                                        RNN_training(X_windows,Y_windows,nr_of_features,nr_of_outputs,model_choice,epochs,batch_size,seq_len,pred_len,learning_rate,nr_of_hidden_neurons)



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
