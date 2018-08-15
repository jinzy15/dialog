from BaseCooc import BaseCooc
from snownlp import SnowNLP
class SnowCooc(BaseCooc):
    def filtseq(self,seq):
        line = seq.strip().encode().decode('utf-8', 'ignore')
        if len(line)==0:
            return ''
        s = SnowNLP(line)
        remain = ['n', 'd', 'v']
        res = []
        for i, j in s.tags:
            if j[0] in remain:
                res.append(i)
        return ' '.join(res)

basecooc = SnowCooc()
# basecooc.transet('../data/all_set_add.set','all_snow','all_snow')
basecooc.load('all_snow','all_snow')

print(basecooc.search('哎,你忙吧'))
import ipdb;ipdb.set_trace()