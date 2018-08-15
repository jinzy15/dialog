#! /usr/local/python3.6.5/bin/python3.6
# -*- coding: utf-8 -*-
import sys
sys.path.append('../tfidf/tfidf_')
sys.path.append('../dataprocess/')
sys.path.append('../cooccurre/')
from AddRank import *
from HredRank import *
from TfidfRank import *
from EncodeRank import *
from HredEncodeRank import *
from AddRank import *
from fileObject import *
from bleuRank import *
from dataprocess.unit import *
from cooccurre.BaseCooc import BaseCooc
from cooccurre.SnowCooc import SnowCooc
import numpy as np
from dataprocess.processor import Processor

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex

def minussoftmax(x):
    ex = np.exp(-np.array(x))
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex

def pick30(sentence):
    tmp = sentence.split(' ')
    tmp = tmp[-29:]
    tmp = ' '.join(tmp)
    return tmp

class mergeselect(object):
    def __init__(self):
        abs_file = os.path.dirname(__file__)
        self.baserank = BaseRank()
        # self.baserank.set(abs_file + '/../data/allchat.set')
        self.hredrank = HredRank()
        # self.hredrank.set(abs_file + '/../data/allchat.set')

        # file_obj = FileObj(abs_file + "/../tfidf/tfidf_data/context.txt")
        # train_sentences = file_obj.read_lines()
        # file_obj = FileObj(abs_file + "/../tfidf/tfidf_data/questions.txt")
        # q_train_sentences = file_obj.read_lines()
        # file_obj = FileObj(abs_file + "/../tfidf/tfidf_data/answers.txt")
        # answer = file_obj.read_lines()

        # self.tfidfrank = TfidfRank(train_sentences, q_train_sentences, answer)
        # self.tfidfrank.set(abs_file + '/../data/allchat.set')

        self.hredencoderank = HredEncodeRank()
        # self.hredencoderank.set(abs_file + '/../data/allchat.set')
        self.encoderank = EncodeRank()
        # self.encoderank.set(abs_file + '/../data/allchat.set')

        self.addrank = AddRank()
        self.addrank.set(abs_file + '/../data/AddRank.set')

        self.bleuRank = bleuRank()
        self.basecooc = BaseCooc()
        self.basecooc.load('../cooccurre/all_nor', '../cooccurre/all_nor')
        self.snowcooc = SnowCooc()
        self.snowcooc.load('../cooccurre/all_snow', '../cooccurre/all_snow')

    def get_add_features(self,input,n):

        unit = input
        top_n = self.addrank.search(unit,n,ascending=False)
        add_features = [output[0] for output in top_n]
        outputs = [output[-1] for output in top_n]
        unitset = UnitSet()
        # for output in outputs:
        #     tmp = Unit()
        #     # sen = [pick30(ch_normalizeAString(s)) for s in top]
        #     tmp.context = output
        #     unitset.allunit.append(tmp)
        unitset.allunit = outputs
        processor = Processor()
        unit.context = [pick30(ch_normalizeAString(s)) for s in unit.context]
        # unitset = processor.run(unitset=unitset)

        # hredranks = softmax([self.hredrank.distance(unit, t) for t in unitset])
        # baseranks = minussoftmax([self.baserank.distance(unit, t) for t in unitset])
        # encoderanks = softmax([self.encoderank.distance(unit, t) for t in unitset])
        # addranks = softmax(add_features)
        # hredencoderanks = softmax([self.hredencoderank.distance(unit, t) for t in unitset])

        baseranks = [self.baserank.distance(unit, t) for t in unitset]
        encoderanks = [self.encoderank.distance(unit, t) for t in unitset]
        addranks = add_features
        hredencoderanks = [self.hredencoderank.distance(unit, t) for t in unitset]
        bleuRanks =  [self.bleuRank.distance(unit, t) for t in unitset]
        basecoocs = [self.basecooc.cooccurre(unit.context[-1],t.context[-1]) for t in unitset]

        # snowranks = softmax([snowrank.cooccurre(unit.context[-1], t.context[-1]) for t in unitset])
        # import ipdb;ipdb.set_trace()

        features = [list(ranks) for ranks in zip(baseranks,encoderanks,hredencoderanks,addranks,bleuRanks,basecoocs)]
        outputs = [output.context[-1] for output in outputs]
        return outputs,features

    ## input is a session(type:unit) with last answer
    ## n is number of outputs and features you return
    ## ouputs is using merge rank to pick best n answers from context session(string) [answer1,answer2.....]
    ## features is all the feature we have now(type numpy.array) (range 0-1) [answer1_features1,answer_features2.....]

    def get_features(self,input,n):
        #your code here
        unit = input
        top_n = self.tfidfrank.search(input,n)
        outputs = [output[-1] for output in top_n]
        # features are ordered by (baserank,)
        unitset = UnitSet()

        for top in top_n:
            tmp = Unit()
            sen = [pick30(ch_normalizeAString(s)) for s in top]
            tmp.context = sen
            unitset.allunit.append(tmp)

        processor = Processor()

        unit.context = [pick30(ch_normalizeAString(s)) for s in unit.context]
        unitset = processor.run(unitset=unitset)
        hredranks = softmax([self.hredrank.distance(unit, t) for t in unitset])
        baseranks = minussoftmax([self.baserank.distance(unit, t) for t in unitset])
        encoderanks = softmax([self.encoderank.distance(unit, t) for t in unitset])
        addranks = softmax([self.addrank.distance(unit, t) for t in unitset])
        # snowranks = softmax([snowrank.cooccurre(unit.context[-1], t.context[-1]) for t in unitset])
        hredencoderanks = softmax([self.hredencoderank.distance(unit, t) for t in unitset])
        features = [list(ranks) for ranks in zip(baseranks,hredranks,encoderanks,hredencoderanks,addranks)]
        return outputs,features

    def mergeSelect(self,unit):
        top20 = self.tfidfrank.search(unit)
        unitset = UnitSet()
        for top in top20:
            tmp = Unit()
            sen = [pick30(ch_normalizeAString(s)) for s in top]
            tmp.context = sen
            unitset.allunit.append(tmp)
        processor = Processor()
        unit.context = [pick30(ch_normalizeAString(s)) for s in unit.context]
        unitset = processor.run(unitset=unitset)
        hredranks = softmax([self.hredrank.distance(unit, t) for t in unitset])
        baseranks = minussoftmax([self.baserank.distance(unit, t) for t in unitset])
        encoderanks = softmax([self.encoderank.distance(unit, t) for t in unitset])
        addranks = softmax([self.addrank.distance(unit, t) for t in unitset])
        # snowranks = softmax([snowrank.cooccurre(unit.context[-1], t.context[-1]) for t in unitset])
        hredencoderanks = softmax([self.hredencoderank.distance(unit, t) for t in unitset])
        mergeranks = [sum(ranks) for ranks in zip(hredranks, baseranks, encoderanks, hredencoderanks,addranks)]
        sorted_merged_ranks = sorted(list(enumerate(mergeranks)), key=lambda item: item[1], reverse=True)
        top_index = sorted_merged_ranks[0][0]
        answer = top20[top_index][-1]
        return answer

    def run(self,input_file_path,output_file_path):
        file_obj = FileObj(input_file_path)
        test_sentences = file_obj.read_lines()
        file_result = open(output_file_path, 'w')
        for sentence in test_sentences:
            unit = Unit()
            unit.context = sentence.split('<s>')
            result = self.mergeSelect(unit)+'\n'
            file_result.write(result)
