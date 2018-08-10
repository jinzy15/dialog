import sys
sys.path.append('../')
sys.path.append('../dataprocess')

from dataprocess.unit import *

class BaseRank(object):
    def __init__(self):
        self.unitset = None
    def distance(self,s1,s2):
        first = s1.context[-1]
        try:
            second = s2.context[-2]
            if len(first) > len(second):
                first, second = second, first
            if len(second) == 0:
                return len(first)
            first_length = len(first) + 1
            second_length = len(second) + 1
            distance_matrix = [[0] * second_length for x in range(first_length)]
            for i in range(first_length):
                distance_matrix[i][0] = i
            for j in range(second_length):
                distance_matrix[0][j] = j
            for i in range(1, first_length):
                for j in range(1, second_length):
                    deletion = distance_matrix[i - 1][j] + 1
                    insertion = distance_matrix[i][j - 1] + 1
                    substitution = distance_matrix[i - 1][j - 1]
                    if first[i - 1] != second[j - 1]:
                        substitution += 1
                    distance_matrix[i][j] = min(insertion, deletion, substitution)
            return distance_matrix[first_length - 1][second_length - 1]
        except:
            print('lenght have problem')
            return 0
    def score(self,s,unitset=None):
        if(unitset==None):
            unitset = self.unitset
        assert unitset!=None,'have to initialize unitset or pass one'
        return [self.distance(s,unit) for unit in unitset]

    def rerank(self,s,ascending=True,unitset=None):
        if(unitset==None):
            unitset = self.unitset
        assert unitset!=None,'have to initialize unitset or pass one'
        temp = [(self.distance(s,unit),unit) for unit in unitset]
        if(ascending):
            return sorted(temp, key=lambda unit : unit[0])
        else:
            return sorted(temp, key=lambda unit: -unit[0])

    def search(self,s,kmax = 10,ascending=True,unitset=None):
        if(unitset==None):
            unitset = self.unitset
        assert unitset!=None,'have to initialize unitset or pass one'
        res = self.rerank(s,ascending,unitset)
        return res[:kmax]

    def set(self,unitset_path):
        a = UnitSet()
        a.load(unitset_path)
        self.unitset = a
    def save(self):
        pass
    def load(self):
        pass

# baserank = BaseRank()
# baserank.set('../data/five_set.set')
#
# s = baserank.unitset[0]
# print(s.context)
# print(baserank.search(s))


