import pandas as pd
import numpy as np
import pickle as pkl
import sys
sys.path.append('../')
# from Config import *

class Unit(object):
    def __init__(self):
        self.context = []
        self.user = None
        self.session_id = None
        self.order_of_session = []
    def __repr__(self):
        return str(self.context)
    def __getitem__(self, item):
        return self.context[item]
    def __len__(self):
        return len(self.context)

class UnitSet(object):
    def __init__(self):
        self.allunit = []

    def __getitem__(self, key):
        return self.allunit[key]

    def __len__(self):
        return int(len(self.allunit))

    def load_df(self,path):
        df = pd.read_csv(path)
        [['session_id', 'user_id', 'send', 'zhaunchu',
          'repeat', 'sku', 'content']]
        for idx, item in df.groupby(df.session_id):
            if (len(idx) > 1):
                temp,flag = self.df2unit(idx, item)
                if(flag):
                    self.allunit.append(temp)
    def df2unit(self,session_id,item):
        try:
            myunit = Unit()
            myunit.session_id = session_id
            myunit.user = item.iloc[0].user_id
            for i in range(len(item)):
                myunit.order_of_session.append(int(item.iloc[i].send))
                myunit.context.append(item.iloc[i].content)
            return myunit,True
        except:
            return None,False
            pass
    def unit2txt(self,path):
        all_list = []
        for item in self.allunit:
            temp_context = list(map(lambda x:"<SOS> "+x+" <EOS>",item.context))
            all_list.append(' '.join(temp_context))
        all_str = str('\n'.join(all_list))
        print(type(all_str))
        f = open(path,'w')
        f.write(all_str)
        f.close()

    def save(self,path):
        pkl.dump(self.allunit,open(path,'wb'))
    def load(self, path):
        self.allunit = pkl.load(open(path,'rb'))

#
# unitset = UnitSet()
# unitset.load_df('../data/allchat.csv')
# unitset.save('../data/allchat.set')