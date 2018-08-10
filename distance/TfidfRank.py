import sys
sys.path.append("../")
sys.path.append("../tfidf")
sys.path.append("../tfidf/tfidf_/")
sys.path.append("../tfidf/tftdf_data/")
from fileObject import FileObj
from sentenceSimilarity import SentenceSimilarity
from sentence import Sentence
from sklearn.externals import joblib
from cutWords import *
from BaseRank import *
sys.path.append('../dataprocess')
from dataprocess.unit import *
from BaseRank import *

class TfidfRank(BaseRank):
    def __init__(self,train_sentences,q_train_sentences,answer):
        self.unitset = None
        self.context = train_sentences
        self.answer = answer
        self.question = q_train_sentences
        self.seg = Seg()
        self.ss = SentenceSimilarity(self.seg)
        self.ss.set_sentences(train_sentences)
        self.ss.TfidfModel()
        self.qss = SentenceSimilarity(self.seg)
        self.qss.set_sentences(self.question)
        self.qss.TfidfModel()

    def distance(self,s1,s2):
        s1 =','.join(s1.context[:-1])
        s2 = ','.join(s2.context[:-1])
        sim = self.ss.sensimilarity(s1,s2)
        return sim

    def search(self,s,n):
        q_sims = self.qss.similarity(s.context[-2])
        context_sims = self.ss.similarity(' '.join(s.context[:-2]))
        sims = [sum(sim) for sim in zip(q_sims,context_sims)]
        sims = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        con_sorted_sims = [self.context[sim[0]].split(',') for sim in sims][:n]
        q_sorted_sims = [[self.question[sim[0]]] for sim in sims][:n]
        a_sorted_sims = [[self.answer[sim[0]]] for sim in sims][:n]
        sorted_sim = [sim[0]+sim[1]+sim[2] for sim in zip(con_sorted_sims,q_sorted_sims,a_sorted_sims)]
        return sorted_sim
#
# file_obj = FileObj(r"../tfidf/tfidf_data/context.txt")
# train_sentences = file_obj.read_lines()
# baserank = TfidfRank(train_sentences)
# baserank.set('../data/five_set.set')
# s = baserank.unitset[0]
# result = baserank.search(s)
# for r in result:
#     print(r)

