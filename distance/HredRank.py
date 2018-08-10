import sys
sys.path.append('../glove_hred')
from BaseRank import *
from glove_hred.all_package import *

class HredRank(BaseRank):
    def __init__(self):
        self.unitset = None
        model_path = '../glove_hred/'
        word2vec = KeyedVectors.load_word2vec_format(glove_data)
        mydata = hredVectorchSet('../'+hredVectorchSet_name + '.set', '../'+hredVectorchSet_name + '.lang', word2vec)
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
        print(s1)
        print(s2)
        return self.mymodel.score(s1.context,s2.context[-1],MAX_LENGTH)


# baserank = HredRank()
#
# baserank.set('../data/five_set.set')
# #
# s = baserank.unitset[0]
# print(s.context)
# check = baserank.unitset[:10]
# print(baserank.search(s,ascending=False))
