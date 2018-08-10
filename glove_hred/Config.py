import torch
# import matplotlib.pyplot as plt

# plt.switch_backend('agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
description = 'hred_debug_mode'
is_train = True
is_evaluate = True
use_histmodel = True
histmodel_path = 'train_fruit/'+description
pair_file = 'data/chat_pair.txt'
numpy_file = 'data/clean_group.npy'
lang_file = './utils/npLang'
first_use_lang = True

glove_data = '../glove_data/word2vec.txt'
hredVectorchSet_name =  'data/glove_set_divide'

teacher_forcing_ratio = 0.9
learning_rate=0.001
hidden_size = 256
EOS_token = 2
SOS_token = 1
PAD_token = 0
UNK_token = 3
batch_size = 30
MAX_LENGTH = 30