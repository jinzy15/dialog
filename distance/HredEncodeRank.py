import sys
sys.path.append('../glove_hred')
from BaseRank import *
from glove_hred.all_package import *
import os

class HredEncodeRank(BaseRank):
    def __init__(self):
        self.unitset = None
        abs_file = os.path.dirname(__file__)+'/'
        model_path = abs_file+'../glove_hred/'
        word2vec = KeyedVectors.load_word2vec_format(abs_file+glove_data)
        mydata = hredVectorchSet(abs_file+'../'+hredVectorchSet_name + '.set', abs_file+'../'+hredVectorchSet_name + '.lang', word2vec)
        input_size = mydata.lang.n_words
        output_size = mydata.lang.n_words
        trainloader = DataLoader(mydata)
        testloader = DataLoader(mydata)
        encoder = Modules.gloveEncoderRNN(input_size, hidden_size).to(device)
        decoder = Modules.HREDAttnDecoderRNN(hidden_size, output_size).to(device)
        context = Modules.ContextRNN(hidden_size, hidden_size).to(device)
        histmodel_path = 'train_fruit/hred_debug_mode'
        encoder.load_state_dict(
            torch.load(model_path + histmodel_path + '_encoder.pkl', map_location=lambda storage, loc: storage))
        decoder.load_state_dict(
            torch.load(model_path + histmodel_path + '_decoder.pkl', map_location=lambda storage, loc: storage))
        context.load_state_dict(
            torch.load(model_path + histmodel_path + '_context.pkl', map_location=lambda storage, loc: storage))
        mymodel = gloveHRED(mydata, trainloader, testloader, encoder, decoder, context)
        self.mymodel = mymodel

    def distance(self, s1, s2):
        o1 = self.mymodel.encodescore(s1.context).cpu().detach().numpy()[0,0]
        o2 = self.mymodel.encodescore(s2.context[:-1]).cpu().detach().numpy()[0,0]
        num = float(np.dot(o1,o2))  # 若为行向量则 A * B.T
        denom = np.linalg.norm(o1) * np.linalg.norm(o2)
        cos = num / denom  # 余弦值
        sim = 0.5 + 0.5 * cos  # 归一化
        return sim

# baserank = HredEncodeRank()
# baserank.set('../data/five_set.set')
# s = baserank.unitset[0]
# print(s.context)
# check = baserank.unitset[:10]
# print(baserank.search(s,ascending=False))
