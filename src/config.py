
"""Training config"""
max_epochs = 1000
test_ratio = 0.2
val_ratio = 0.2
patience = 10

"""RNN config"""
batch_size_list =  [100]
seq_len_list = [20]
pred_len_list = [5]
nr_of_hidden_neurons_list = [5]
model_choice_list = ["GRU"]
learning_rate_list = [1e-3]

