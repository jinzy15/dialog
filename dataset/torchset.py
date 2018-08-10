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
# from Lang import gloveLang
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
        return [self.lang.word2index.get(word,3) for word in sentence.split(' ')]

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


class VectorchSet(Dataset):
    def __init__(self,unitset_path,lang_path,word2vec):
        self.unitset = UnitSet()
        self.word2vec = word2vec
        self.unitset.load(unitset_path)
        self.lang = unitLang()
        self.lang.loadLang(lang_path)
        # self.batch_size = batch_size
        self.testset = self.unitset[int(len(self.unitset)*0.9):]
        self.unitset = self.unitset[0:int(len(self.unitset)*0.9)]

    def __getitem__(self, index):
        return self.tensorsFromSession(self.unitset[index].context)

    def __len__(self):
        return len(self.unitset)

    def vecFromSentence(self, sentence):
        return [self.vecFromword(word) for word in sentence.split(' ')]

    def indexesFromSentence(self, sentence):
        return [self.lang.word2index.get(word,3) for word in sentence.split(' ')]

    def vecFromword(self,word):
        # print(word)
        if (word in self.word2vec.vocab):
            return self.word2vec.get_vector(word)
        else:
            return self.word2vec.get_vector('<UNK>')


    def tensorFromSentence(self, sentence):
        indexes = self.vecFromSentence(sentence)
        indexes.append(self.word2vec.get_vector('<EOS>'))
        # print(indexes)
        return torch.tensor(indexes, device=device)

    def targetFromSentence(self, sentence):
        indexes = self.indexesFromSentence( sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromSession(self,session):
        temp = []
        for sentence in session[:-1]:
            tmp = self.tensorFromSentence(sentence)
            temp.append(tmp)
        temp.append(self.targetFromSentence(session[-1]))
        return temp

#对话同时返回向量和id
class hredVectorchSet(Dataset):
    def __init__(self,unitset_path,lang_path,word2vec):
        self.unitset = UnitSet()
        self.word2vec = word2vec
        self.unitset.load(unitset_path)
        self.lang = unitLang()
        self.lang.loadLang(lang_path)
        self.testset = self.unitset[int(len(self.unitset)*0.9):]
        self.unitset = self.unitset[0:int(len(self.unitset)*0.9)]

    def __getitem__(self, index):
        return self.tensorsFromSession(self.unitset[index].context)

    def __len__(self):
        return len(self.unitset)

    def vecFromSentence(self, sentence):
        return [self.vecFromword(word) for word in sentence.split(' ')]

    def idFromSentence(self, sentence):
        return [self.lang.word2index.get(word,3) for word in sentence.split(' ')]

    def vecFromword(self,word):
        if (word in self.word2vec.vocab):
            return self.word2vec.get_vector(word)
        else:
            return self.word2vec.get_vector('<UNK>')

    def vectensorFromSentence(self, sentence):
        indexes = self.vecFromSentence(sentence)
        indexes.append(self.word2vec.get_vector('<EOS>'))
        return torch.tensor(indexes, device=device)

    def idtensorFromSentence(self, sentence):
        indexes = self.idFromSentence(sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def idtensorsFromSession(self,session):
        temp = []
        for sentence in session:
            tmp = self.idtensorFromSentence(sentence)
            temp.append(tmp)
        return temp

    def vectensorsFromSession(self,session):
        temp = []
        for sentence in session:
            tmp = self.vectensorFromSentence(sentence)
            temp.append(tmp)
        return temp

    def tensorsFromSession(self, session):
        return self.vectensorsFromSession(session),self.idtensorsFromSession(session)