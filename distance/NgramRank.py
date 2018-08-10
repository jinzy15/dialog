import sys
sys.path.append('../')
sys.path.append('../dataprocess')
from dataprocess.unit import *

class NgramRank(BaseRank):
    def __init__(self,N=3):
        self.N = N
    def distance(self,s1,s2): #s1 is a query session s2 is session pool
        def D(i,j,N):

        first = s1.context[-1]
        try:
            second = s2.context[-2]





