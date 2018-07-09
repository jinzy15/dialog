from abc import ABCMeta, abstractmethod
import random
import torch
import time
from Config import *
from torch import optim
import torch.nn as nn
from utils.Timer import timeSince
import pickle as pkl
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

class torchsetModels(object):
    __metaclass__ = ABCMeta
    def __init__(self,train_loader,test_loader,**kw):
        pass
    @abstractmethod
    def train(self,input_tensor):
        pass
    @abstractmethod
    def trainEpoch(self):
        pass
    @abstractmethod
    def evaluate(self,input):
        pass

    def evaluateRandomly(self,n):
        for i in range(n):
            session = random.choice(self.dataset.unitset)
            session = session.context
            print('>', session[0:-1])
            print('=', session[-1])
            output_words = self.evaluate(session[0:-1],MAX_LENGTH)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
            print('')

    

