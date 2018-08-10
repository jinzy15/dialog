from all_package import *

word2vec = KeyedVectors.load_word2vec_format(glove_data)
# mydata = hredVectorchSet('data/glove_set_divide.set','data/glove_set_divide.lang',word2vec)
mydata = hredVectorchSet(hredVectorchSet_name+'.set',hredVectorchSet_name+'.lang',word2vec)

input_size = mydata.lang.n_words
output_size = mydata.lang.n_words

trainloader = DataLoader(mydata)
testloader = DataLoader(mydata)

encoder = Modules.gloveEncoderRNN(input_size, hidden_size).to(device)
decoder = Modules.HREDAttnDecoderRNN(hidden_size, output_size).to(device)
context = Modules.ContextRNN(hidden_size,hidden_size).to(device)

if (use_histmodel):
    encoder.load_state_dict(torch.load(histmodel_path+'_encoder.pkl'))
    decoder.load_state_dict(torch.load(histmodel_path+'_decoder.pkl'))
    context.load_state_dict(torch.load(histmodel_path+'_context.pkl'))

mymodel = gloveHRED(mydata,trainloader,testloader,encoder,decoder,context)

if(is_train):
    mymodel.trainEpoch(10,100,learning_rate=learning_rate, plot_every=100)

if(is_evaluate):
    mymodel.evaluateRandomly(50)


