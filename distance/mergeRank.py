#! /usr/local/python3.6.5/bin/python3.6
# -*- coding: utf-8 -*-
import sys
sys.path.append('../tfidf/tfidf_')
sys.path.append('../dataprocess/')
from HredRank import *
from TfidfRank import *
from EncodeRank import *
from HredEncodeRank import *
from AddRank import *
from fileObject import *
from unit import *
import numpy as np
from dataprocess.processor import Processor

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

def minussoftmax(x):
    ex = np.exp(-np.array(x))
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

def pick30(sentence):
    tmp = sentence.split(' ')
    tmp = tmp[-29:]
    tmp = ' '.join(tmp)
    return tmp

def run(input_file_path,output_file_path):
    abs_file = os.path.dirname(__file__) + '/'
    def mergeSelect(unit):
        top20 = tfidfrank.search(unit)
        unitset = UnitSet()
        for top in top20:
            tmp = Unit()
            sen = [pick30(ch_normalizeAString(s)) for s in top]
            tmp.context = sen
            unitset.allunit.append(tmp)
        processor = Processor()
        unit.context = [pick30(ch_normalizeAString(s)) for s in unit.context]
        unitset = processor.run(unitset=unitset)
        hredranks = softmax([hredrank.distance(unit, t) for t in unitset])
        baseranks = minussoftmax([baserank.distance(unit, t) for t in unitset])
        encoderanks = softmax([encoderank.distance(unit, t) for t in unitset])
        addranks = softmax([addrank.distance(unit, t) for t in unitset])
        # snowranks = softmax([snowrank.cooccurre(unit.context[-1], t.context[-1]) for t in unitset])
        hredencoderanks = softmax([hredencoderank.distance(unit, t) for t in unitset])
        mergeranks = [sum(ranks) for ranks in zip(hredranks, baseranks, encoderanks, hredencoderanks,addranks)]
        sorted_merged_ranks = sorted(list(enumerate(mergeranks)), key=lambda item: item[1], reverse=True)
        top_index = sorted_merged_ranks[0][0]
        answer = top20[top_index][-1]
        print(answer)
        return answer

    abs_file = os.path.dirname(__file__)
    baserank = BaseRank()
    baserank.set(abs_file+'/../data/allchat.set')

    hredrank = HredRank()
    hredrank.set(abs_file+'/../data/allchat.set')

    file_obj = FileObj(abs_file+"/../tfidf/tfidf_data/context.txt")
    train_sentences = file_obj.read_lines()
    file_obj = FileObj(abs_file + "/../tfidf/tfidf_data/questions.txt")
    q_train_sentences = file_obj.read_lines()
    file_obj = FileObj(abs_file + "/../tfidf/tfidf_data/answers.txt")
    answer = file_obj.read_lines()

    tfidfrank = TfidfRank(train_sentences,q_train_sentences,answer)
    tfidfrank.set(abs_file+'/../data/allchat.set')

    hredencoderank = HredEncodeRank()
    hredencoderank.set(abs_file+'/../data/allchat.set')
    encoderank = EncodeRank()
    encoderank.set(abs_file+'/../data/allchat.set')
    addrank = AddRank()
    addrank.set(abs_file+'/../data/allchat.set')

    file_obj = FileObj(input_file_path)
    test_sentences = file_obj.read_lines()
    file_result = open(output_file_path, 'w')
    for sentence in test_sentences:
        unit = Unit()
        unit.context = sentence.split('<s>')
        result = mergeSelect(unit)+'\n'
        file_result.write(result)

