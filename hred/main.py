import torch
import random
import transformer.Modules as Modules
from torch import optim
import utils.Lang as Lang
from dataset.torchset import *
import torch.nn as nn
from Config import *
import transformer.Modules as Modules
from transformer.Models.AHRED import AHRED
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from gensim.models import KeyedVectors
#
# word2vec = KeyedVectors.load_word2vec_format('../glove_data/word2vec.txt')
# mydata = VectorchSet('data/hred_set.set','data/hred_set.lang',word2vec)

mydata = TorchSet('data/hred_set.set', 'data/hred_set.lang')
input_size = mydata.lang.n_words
output_size = mydata.lang.n_words

trainloader = DataLoader(mydata)
testloader = DataLoader(mydata)

encoder = Modules.EncoderRNN(input_size, hidden_size).to(device)
decoder = Modules.HREDAttnDecoderRNN(hidden_size, output_size).to(device)
context = Modules.ContextRNN(hidden_size,hidden_size).to(device)

if (use_histmodel):
    encoder.load_state_dict(torch.load(histmodel_path+'_encoder.pkl'))
    decoder.load_state_dict(torch.load(histmodel_path+'_decoder.pkl'))
    context.load_state_dict(torch.load(histmodel_path+'_context.pkl'))

mymodel = AHRED(mydata,trainloader,testloader,encoder,decoder,context)

if(is_train):
    mymodel.trainEpoch(10,100,learning_rate=learning_rate,plot_every=10)
    # mymodel.trainEpoch(50, 100, learning_rate=learning_rate)
    # mymodel.trainIters(10)
if(is_evaluate):
    mymodel.evaluateRandomly(50)


