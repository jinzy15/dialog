import torch
num_epoch = 1
batch_size = 5

features_num = 8
EOS_token = 2
SOS_token = 1
PAD_token = 0
UNK_token = 3
batch_size = 30
MAX_LENGTH = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")