import re
import jieba


def ch_normalizeAString(s):
    s = s.encode().decode("utf8")
    s = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——《》【】：”“！-，。？?、~@#￥%……&*（）]+".encode().decode("utf8"),
               "".encode().decode("utf8"), s)
    s = translate(s)
    s = ' '.join(list(jieba.cut(s, cut_all=False)))
    return s


def translate(str):
    line = str.strip().encode().decode('utf-8', 'ignore')
    p2 = re.compile(u'[^\u4e00-\u9fa5]')
    zh = "".join(p2.split(line)).strip()
    zh = "".join(zh.split())
    outStr = zh  # 经过相关处理后得到中文的文本
    return outStr


def ch_normalizeString(item):
    rtn = []
    for s in item:
        if (type(s) == str):
            s = s.encode().decode("utf8")
            s = re.sub("[A-Za-z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——《》【】：”“！-，。？?、~@#￥%……&*（）]+:".encode().decode("utf8"),
                       "".encode().decode("utf8"), s)
            s = self.translate(s)
            s = ' '.join(list(jieba.cut(s, cut_all=False)))
            rtn.append(s)
    return rtn
