from picker import *
from processor import *
from formater import *
from unit import *
from utils.Lang import unitLang
#
class Pipe:
    def __init__(self,load_path,to_path):
        self.load_path = load_path
        self.to_path = to_path
        self.filters = []
    def run(self):
        myset = UnitSet()
        myset.load(self.load_path)
        for filter in self.filters:
            myset = filter.run(myset)
        lang = unitLang()
        lang.prepareData(myset)
        lang.saveLang(self.to_path+'.lang')
        myset.save(self.to_path+'.set')

processor = Processor()  #对文本进行分词,和过滤
addpro = addProcessor()  #对文本顺序进行合成
lastPick = pickLastProcessor() #只选取最后一轮对话
lenpicker = sentencelenPicker(min=-1,max=30) #选择句子的长度
slenpicker =sesslenPicker(min=2)  #选择对话的轮数
slenpicker2 = sesslenPicker( min=2, max = 6)

# five_set_list = [addpro,processor,slenpicker2,lenpicker]
last_seq_list = [addpro,processor,lastPick,slenpicker,lenpicker]
mypipe = Pipe('../data/chat.pkl','../data/last_set')
mypipe.filters = last_seq_list
# mypipe = Pipe('../data/chat.pkl','../data/five_set')
# mypipe.filters = five_set_list
mypipe.run()


