import torch


def split_data(all_data, ratio):
    n = len(all_data)  # total number of examples
    n_val = int(ratio * n)  # take ~10% for val
    train_data, val_data = torch.utils.data.random_split(all_data, [(n - n_val), n_val])

    return train_data, val_data