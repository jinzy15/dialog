import sys
sys.path.append('../glove_hred')
from BaseRank import *
from glove_hred.all_package import *
import os

class HredRank(BaseRank):
    def __init__(self):
        self.unitset = None
        abs_file = os.path.dirname(__file__) + '/'
        model_path = abs_file+'../glove_hred/'
        glove_data = '../glove_data/word2vec.txt'
        word2vec = KeyedVectors.load_word2vec_format(abs_file+glove_data)
        mydata = hredVectorchSet(abs_file+'../'+'data/glove_set_divide' + '.set',
                                 abs_file+'../'+'data/glove_set_divide' + '.lang',
                                 word2vec)
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
        return self.mymodel.score(s1.context,s2.context[-1],MAX_LENGTH)


# baserank = HredRank()
#
# baserank.set('../data/five_set.set')
# #
# s = baserank.unitset[0]
# print(s.context)
# check = baserank.unitset[:10]
# print(baserank.search(s,ascending=False))
