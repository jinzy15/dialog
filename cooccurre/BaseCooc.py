import sys
sys.path.append('../')
sys.path.append('../dataprocess')
from dataprocess.unit import *
import jieba
import re
import json
import pickle
#every set if want to transed by cooc need be piped from addprocessor first

class BaseCooc(object):
    def __init__(self):
        self.q_bag = None
        self.a_bag = None
        self.S = None

    def load(self,bag_name,json_name):
        q_bag_in = open(bag_name+'.qbag','rb')
        a_bag_in = open(bag_name + '.abag', 'rb')
        S_in = open(json_name+'.S','rb')
        self.q_bag = pickle.load(q_bag_in)
        self.a_bag = pickle.load(a_bag_in)
        self.S = pickle.load(S_in)
        q_bag_in.close()
        a_bag_in.close()
        S_in.close()

    def search(self,query):
        query = self.filtseq(query)
        return self.S.get(query,-1)

    def cooccurre(self,query,answer):
        query = self.filtseq(query)
        answer = self.filtseq(answer)
        if query in self.S:
            temp = self.S[query]
            if(answer in temp):
                all = float(sum(len(i) for i in temp.values()))
                return len(temp[answer])/all
            else:
                return 0
        else:
            return 0

    def transet(self,set_path,bag_name,json_name):
        myset = UnitSet()
        myset.load(set_path)
        q_bag,a_bag = {},{}
        address =0
        S = {}

        for session in myset:
            content = session.context
            for i in range(0,len(content),2):
                # import ipdb;ipdb.set_trace()
                query = content[i]
                if(i+1<len(content)):
                    answer = content[i+1]
                else:
                    answer = ''
                q_bag[address] = query
                a_bag[address] = answer
                Q = self.filtseq(query)
                A = self.filtseq(answer)
                # print(Q)
                S[Q] = self.calculate(S.get(Q,0),A,address)
                address += 1
        q_bag_out = open(bag_name+'.qbag','wb')
        a_bag_out = open(bag_name + '.abag', 'wb')
        S_out = open(json_name+'.S','wb')
        pickle.dump(q_bag, q_bag_out, 0)
        pickle.dump(a_bag, a_bag_out, 0)
        pickle.dump(S,S_out,0)
        q_bag_out.close()
        a_bag_out.close()
        S_out.close()

    def calculate(self,res,ans,address):
        # print(res)
        if(res==0):
            return {ans:[address]}
        else:
            if ans in res:
                res[ans].append(address)
                return res
            else:
                res[ans] = [address]
                return res

    def filtseq(self,seq):
        line = seq.strip().encode().decode('utf-8', 'ignore')
        p2 = re.compile(u'[^\u4e00-\u9fa5]')
        zh = "".join(p2.split(line)).strip()
        zh = "".join(zh.split())
        outStr = zh
        outStr = ' '.join(list(jieba.cut(outStr, cut_all=False)))
        if(outStr==None):
            return ''
        return outStr

basecooc = BaseCooc()
# # basecooc.transet('../data/all_set_add.set','all_nor','all_nor')
basecooc.load('all_nor','all_nor')
import ipdb;ipdb.set_trace()
