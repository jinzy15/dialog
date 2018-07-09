# coding: utf-8
from __future__ import unicode_literals, print_function, division
from Config import *
from dataset.torchset import TorchSet
import transformer.Modules as Modules
from transformer.Models.EDModels import EDModels
from torch.utils.data import DataLoader

from io import open
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
import numpy as np

mydata = TorchSet('data/last_set.set','data/last_set.lang')
input_size = mydata.lang.n_words
output_size = mydata.lang.n_words

encoder1 = Modules.EncoderRNN(input_size, hidden_size).to(device)
attn_decoder1 = Modules.AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)

if (use_histmodel):
    encoder1.load_state_dict(torch.load(histmodel_path+'_encoder.pkl'))
    attn_decoder1.load_state_dict(torch.load(histmodel_path+'_decoder.pkl'))

trainloader = DataLoader(mydata)
testloader = DataLoader(mydata)

mymodel = EDModels(mydata,trainloader,testloader,encoder1,attn_decoder1)

if(is_train):
    mymodel.trainEpoch(5,100)
    # mymodel.trainIters(100)
if(is_evaluate):
    mymodel.evaluateRandomly(10)
    output_words= mymodel.evaluate("你好")
    print(output_words)


