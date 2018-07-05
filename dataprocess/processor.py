import re
import jieba
from unit import *
class Processor(object):
    def __init__(self):
        pass

    def run(self, unitset):
        unitset.allunit = [self.processUnit(unit) for unit in unitset]
        return unitset

    def processUnit(self,unit):
        unit.context = self.ch_normalizeString(unit.context)
        return unit

    def translate(self,str):
        line = str.strip().encode().decode('utf-8', 'ignore')
        p2 = re.compile(u'[^\u4e00-\u9fa5]')
        zh = "".join(p2.split(line)).strip()
        zh = "".join(zh.split())
        outStr = zh  # 经过相关处理后得到中文的文本
        return outStr

    def ch_normalizeString(self,item):
        rtn = []
        for s in item:
            if(type(s)==str):
                s = s.encode().decode("utf8")
                s = re.sub("[A-Za-z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——《》【】：”“！-，。？?、~@#￥%……&*（）]+:".encode().decode("utf8"),
                           "".encode().decode("utf8"), s)
                s = self.translate(s)
                s = ' '.join(list(jieba.cut(s, cut_all=False)))
                rtn.append(s)
        return rtn


class addProcessor(Processor):
    def __init__(self):
        pass

    def processUnit(self, unit):
        unit= self.df2pair(unit)
        return unit

    def df2pair(self,unit):
        send = unit.order_of_session
        content = unit.context
        now_chat = 0
        now_str = ''
        pair = []
        order = []
        for i in range(len(send)):
            if (int(send[i]) == int(now_chat)):
                now_str = now_str + str(content[i])
            else:
                pair.append(now_str)
                order.append(now_chat)
                now_chat = int(send[i])
                now_str = content[i]
            if (i == len(send) - 1):
                pair.append(now_str)
                order.append(now_chat)
        unit.context = pair
        unit.order_of_session = order
        return unit

class pickLastProcessor(Processor):
    def processUnit(self,unit):
        unit.context = unit.context[-2:]
        return unit
