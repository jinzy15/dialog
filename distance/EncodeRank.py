import sys
sys.path.append('../glove_seq2seq')
from BaseRank import *
from glove_seq2seq.all_package import *
import numpy as np

class EncodeRank(BaseRank):
    def __init__(self):
        word2vec = KeyedVectors.load_word2vec_format('../glove_data/word2vec.txt')
        mydata = VectorchSet('../data/last_set.set', '../data/last_set.lang', word2vec)
        input_size = mydata.lang.n_words
        output_size = mydata.lang.n_words
        encoder1 = Modules.gloveEncoderRNN(input_size, hidden_size).to(device)
        attn_decoder1 = Modules.AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)
        model_path = '../glove_seq2seq/'
        histmodel_path = 'train_fruit/first_debug_mode'
        if (use_histmodel):
            if (use_cuda):
                encoder1.load_state_dict(torch.load(model_path+histmodel_path + '_encoder.pkl'))
                attn_decoder1.load_state_dict(torch.load(model_path+histmodel_path + '_decoder.pkl'))
            else:
                encoder1.load_state_dict(torch.load(model_path+histmodel_path + '_encoder.pkl', map_location=lambda storage, loc: storage))
                attn_decoder1.load_state_dict(
                    torch.load(model_path+histmodel_path + '_decoder.pkl', map_location=lambda storage, loc: storage))
        trainloader = DataLoader(mydata)
        testloader = DataLoader(mydata)
        self.mymodel = EDModels(mydata, trainloader, testloader, encoder1, attn_decoder1)

    def distance(self,s1,s2):
        o1 = self.mymodel.score(s1.context[-1]).cpu().numpy()[0,0]
        o2 = self.mymodel.score(s2.context[-2]).cpu().numpy()[0,0]
        num = float(np.dot(o1,o2))  # 若为行向量则 A * B.T
        denom = np.linalg.norm(o1) * np.linalg.norm(o2)
        cos = num / denom  # 余弦值
        sim = 0.5 + 0.5 * cos  # 归一化
        return sim

# baserank = EncoderRank()
# baserank.set('../data/five_set.set')
# s = baserank.unitset[20]
# print(s.context)
# check = baserank.unitset[:10]
# print(baserank.search(s,ascending=False))