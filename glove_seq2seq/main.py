# coding: utf-8
from all_package import *

word2vec = KeyedVectors.load_word2vec_format('../glove_data/word2vec.txt')
mydata = VectorchSet('data/last_set.set','data/last_set.lang',word2vec)

input_size = mydata.lang.n_words
output_size = mydata.lang.n_words

encoder1 = Modules.gloveEncoderRNN(input_size, hidden_size).to(device)
attn_decoder1 = Modules.AttnDecoderRNN(hidden_size, output_size, dropout_p=0.1).to(device)

if (use_histmodel):
    if(use_cuda):
        encoder1.load_state_dict(torch.load(histmodel_path+'_encoder.pkl'))
        attn_decoder1.load_state_dict(torch.load(histmodel_path+'_decoder.pkl'))
    else:
        encoder1.load_state_dict(torch.load(histmodel_path + '_encoder.pkl', map_location=lambda storage, loc: storage))
        attn_decoder1.load_state_dict(torch.load(histmodel_path + '_decoder.pkl', map_location=lambda storage, loc: storage))

trainloader = DataLoader(mydata)
testloader = DataLoader(mydata)

mymodel = EDModels(mydata,trainloader,testloader,encoder1,attn_decoder1)

if(is_train):
    mymodel.trainEpoch(5,100)
if(is_evaluate):
    mymodel.evaluateRandomly(10)
    output_words = mymodel.evaluate(ch_normalizeAString("你好"))
    print(mymodel.evaluate(ch_normalizeAString("什么时候发货")))
    print(mymodel.evaluate(ch_normalizeAString("用的什么快递?")))
    print(mymodel.evaluate(ch_normalizeAString("你们还招人么?")))
    print(mymodel.evaluate(ch_normalizeAString("可以退货么?")))
    print(mymodel.evaluate(ch_normalizeAString("没有")))
    print(mymodel.evaluate(ch_normalizeAString("无理由退怎么算邮费")))
    print(mymodel.evaluate(ch_normalizeAString("怎么申请换货已经跳转到售后服务页面")))

