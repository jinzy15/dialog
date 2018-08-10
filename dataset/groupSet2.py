from torch.utils.data import Dataset
from Config import *
import torch
import numpy as np
import torch

class groupSet2(Dataset):

    def __init__(self,lang,batch_size = 30):

        self.lang = lang
        self.sessions = None
        self.batch_size = batch_size

    def __getitem__(self, index):
        return self.tensorsFromSession(self.sessions[index])

    def __len__(self):
        return len(self.sessions)

    def loadfnp(self,path):
        self.sessions = self.filtSessions(np.load(path)[:1000])

    def indexesFromSentence(self, sentence):
        return [self.lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence( sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1),len(indexes)

    def tensorsFromSession(self,session):
        temp = []
        for sentence in session:
            tmp,length = self.tensorFromSentence(sentence)
            temp.append(tmp)
        return temp,len(session)

    def filtAsession(self,session):
        for sen in session:
            if(len(sen)>MAX_LENGTH):
                return False
        return True

    def filtSessions(self,sessions):
        return [session for session in sessions if self.filtAsession(session)]