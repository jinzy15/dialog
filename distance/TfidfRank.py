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
    def __init__(self,train_sentences):
        self.unitset = None
        self.sentences = train_sentences
        self.seg = Seg()
        self.ss = SentenceSimilarity(self.seg)
        self.ss.set_sentences(train_sentences)
        self.ss.TfidfModel()



    def distance(self,s1,s2):
        s1 =','.join(s1.context[:-1])
        s2 = ','.join(s2.context[:-1])
        sim = self.ss.sensimilarity(s1,s2)
        return sim

    def search(self,s):
        sims = self.ss.similarity(s)
        sims = sorted(list(enumerate(sims)), key=lambda item: item[1], reverse=True)
        sorted_sims = [self.sentences[sim[0]].split() for sim in sims]
        return sorted_sims[:20]
#
# file_obj = FileObj(r"../tfidf/tfidf_data/context.txt")
# train_sentences = file_obj.read_lines()
# baserank = TfidfRank(train_sentences)
# baserank.set('../data/five_set.set')
# s = baserank.unitset[0]
# result = baserank.search(s)
# for r in result:
#     print(r)

