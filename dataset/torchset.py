from torch.utils.data import Dataset
from Config import *
import torch
import numpy as np
import torch
import sys
sys.path.append('../')
sys.path.append('../utils')
sys.path.append('../dataprocess/')
from Lang import unitLang
from dataprocess.unit import *

class TorchSet(Dataset):

    def __init__(self,unitset_path,lang_path,batch_size = 30):
        self.unitset = UnitSet()
        self.lang = unitLang()
        self.lang.loadLang(lang_path)
        self.unitset.load(unitset_path)
        self.batch_size = batch_size

        self.testset = self.unitset[int(len(self.unitset)*0.9):]
        self.unitset = self.unitset[0:int(len(self.unitset)*0.9)]

    def __getitem__(self, index):
        return self.tensorsFromSession(self.unitset[index].context)

    def __len__(self):
        return len(self.unitset)


    def indexesFromSentence(self, sentence):
        return [self.lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence( sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromSession(self,session):
        temp = []
        for sentence in session:
            tmp = self.tensorFromSentence(sentence)
            temp.append(tmp)
        return temp



