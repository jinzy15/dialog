import sklearn
import numpy as np
from mergeRank import mergeselect
import torch as pt
from dataprocess.unit import *
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append('../dataprocess/')
from dataprocess.unit import *
import dataprocess
from dataset.torchset import TorchSet
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import disConfig
import utils.Normalizer as Normalizer
import random
pt.cuda.set_device(0)

class MLP(pt.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = pt.nn.Linear(disConfig.features_num,32)
        self.fc2 = pt.nn.Linear(32, 10)
        self.fc3 = pt.nn.Linear(10, 1)
    def forward(self, din):
        dout = pt.nn.functional.relu(self.fc1(din))
        dout = pt.nn.functional.relu(self.fc2(dout))
        return pt.nn.functional.tanh(self.fc3(dout))

chencherry = SmoothingFunction()
import os
abs_file = os.path.dirname(__file__)+'/'

mydata = UnitSet()
mydata.load('../data/AddRank.set')

myselect = mergeselect()
model = MLP()

if pt.cuda.is_available():
    model = model.cuda()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=disConfig.learning_rate)

train = True
save_path = 'five_set.pkl'
evaluate = True
evaluate_times = 10
if_train_have_answer = True
if_eval_have_answer = True

if (os.path.exists(save_path)):
    model.load_state_dict(pt.load(save_path))

#todo:need to confirm whehter the session contains the answer

if(train):
    for i in range(disConfig.num_epoch):
        for j,item in enumerate(mydata):
            import ipdb;ipdb.set_trace()
            if(if_train_have_answer):
                answer = Normalizer.ch_normalizeAString(item.context[-1])
                item.context = item.context[:-1]
            # optimizer.zero_grad()
            score = []
            outputs,features = myselect.get_add_features(item,disConfig.batch_size)
            print(answer)
            print(outputs)
            print(features)
            # import ipdb;ipdb.set_trace()
            # for output in outputs:
            #     output = Normalizer.ch_normalizeAString(output)
            #     score.append(sentence_bleu([output.split(' ')],answer.split(' '),
            #                               weights=(0.25,0.25,0.25,0.25),
            #                               smoothing_function=chencherry.method1))
            # features = pt.FloatTensor(features)
            # score = pt.FloatTensor(score)
            # if pt.cuda.is_available():
            #     features = features.cuda()
            #     score = score.cuda()
            # pred = model(features)
            # loss = criterion(pred.view(-1),score)
            # loss.backward()
            # optimizer.step()
            # print(loss)
        # pt.save(model.state_dict(), save_path)
#
# if(evaluate):
#     for i in range(evaluate_times):
#         session = random.choice(mydata)
#         if (if_eval_have_answer):
#             session.context = session.context[:-1]
#         print(session.context)
#         outputs, features = myselect.get_add_features(session, disConfig.batch_size)
#         features = pt.FloatTensor(features)
#         if pt.cuda.is_available():
#             features = features.cuda()
#         pred = model(features)
#         pred = pred.cpu().detach().numpy()
#         arg = pred.argsort()
#         print(outputs[int(arg[-1])])

