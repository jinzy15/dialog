from unit import *
class Picker(object):
    def __init__(self):
        pass
    def run(self,unitset):
        unitset.allunit = [unit for unit in unitset if self.ifchoose(unit)]
        return unitset
    def ifchoose(self,unit):
        for sen in unit.context:
            if (len(sen) > MAX_LENGTH):
                return False
        return True

class lengthPicker(Picker):
    def ifchoose(self,unit):
        for sen in unit.context:
            if (len(sen) > MAX_LENGTH):
                return False
        return True

class sesslenPicker(Picker):
    def __init__(self,min= -1 , max = 100000000):
        self.min = min
        self.max = max
    def ifchoose(self,unit):
        if(len(unit.context)>=self.min and len(unit.context)<=self.max):
            return True
        return False

class sentencelenPicker(Picker):
    def __init__(self,min= -1 , max = 100000000):
        self.min = min
        self.max = max
    def ifchoose(self,unit):
        for sen in unit.context:
            if (len(sen) >= self.max or len(sen) <= self.min):
                return False
        return True

