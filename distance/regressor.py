import sklearn
import numpy as np
import mergeRank
import torch as pt

import sys
sys.path.append('../dataprocess/')
from unit import *
import dataprocess
from dataprocess.unit import UnitSet
from dataset.torchset import TorchSet

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = pt.nn.Linear(features_num,32)
        self.fc2 = pt.nn.Linear(32, 10)
        self.fc3 = pt.nn.Linear(10, 1)
    def forward(self, din):
        din = din.view(-1, 28 * 28)
        dout = pt.nn.functional.relu(self.fc1(din))
        dout = pt.nn.functional.relu(self.fc2(dout))
        return pt.nn.functional.tanh(self.fc3(dout))

chencherry = SmoothingFunction()
import os
abs_file = os.path.dirname(__file__)+'/'

mydata = UnitSet()
mydata.load(abs_file+'../data/five_set.set')

def loss


for i in range(num_epoch):
    for j,item in enumerate(mydata):
        answer = item.context[-1]
        loss = []
        outputs,features = mergeRank.get_features(item,batch_size)
        for output in outputs:
            loss.append(sentence_bleu([a.split(' ')],b.split(' '),
                                      weights=(0.25,0.25,0.25,0.25),
                                      smoothing_function=chencherry.method1))


















