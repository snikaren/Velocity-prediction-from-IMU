
"""Training config"""
max_epochs = 5
test_ratio = 0.2
val_ratio = 0.2
patience = 10
optuna_hyperparam_trails = 3

"""RNN config"""
batch_size_list =  [32,64]
seq_len_list = [10]
pred_len_list = [10,50,100]
nr_of_hidden_neurons_list = [5,10]
model_choice_list = ["GRU"]
learning_rate_list = [1e-3]

