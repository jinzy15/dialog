from BaseRank import *
from gensim.models import KeyedVectors
#set passed by AddRank has to be split
import numpy as np
class AddRank(BaseRank,have_answer = False):
    def __init__(self):
        self.word2vec = KeyedVectors.load_word2vec_format('../glove_data/word2vec.txt')
    def distance(self,s1,s2):
        first = s1.context[-1]
        second = s2.context[-2]
        first = first.split(' ')
        second = second.split(' ')
        n_first = len(first)
        n_second = len(second)
        vec_first = np.zeros(256)
        vec_second = np.zeros(256)
        for i in range(n_first):
            try:
                vec_first += self.word2vec.get_vector(first[i])
            except:
                continue
        for j in range(n_second):
            try:
                vec_second += self.word2vec.get_vector(second[j])
            except:
                continue
        vec_first = vec_first/n_first
        vec_second = vec_second/n_second
        num = float(np.dot(vec_first ,vec_second))  # 若为行向量则 A * B.T
        denom = np.linalg.norm(vec_first) * np.linalg.norm(vec_second)
        cos = num / denom  # 余弦值
        sim = 0.5 + 0.5 * cos  # 归一化
        return sim

# baserank = AddRank()
# baserank.set('../data/five_set.set')
# s = baserank.unitset[20]
# print(s.context)
# check = baserank.unitset[:10]
# print(baserank.search(s,ascending=False))