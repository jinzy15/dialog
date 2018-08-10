import sys
sys.path.append('../')
sys.path.append('../dataprocess')
from BaseRank import *
from dataprocess.unit import *
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


class bleuRank(BaseRank):
    def distance(self,s1,s2):
        reference = []
        reference.append(list(s1.context[-2].strip()))
        candidate = list(s2.context[-2].strip())
        chencherry = SmoothingFunction()
        bleuScore = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25),
                                  smoothing_function=chencherry.method1)
        return bleuScore


bluerank = bleuRank()
bluerank.set('../data/five_set.set')

s = bluerank.unitset[0]
print(s.context)
print(bluerank.search(s,ascending=False))