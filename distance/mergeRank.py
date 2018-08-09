#! /usr/local/python3.6.5/bin/python3.6
# -*- coding: utf-8 -*-
import sys
sys.path.append('../tfidf/tfidf_')
sys.path.append('../dataprocess/')
from HredRank import *
from TfidfRank import *
from EncodeRank import *
from HredEncodeRank import *
from fileObject import *
from unit import *
import numpy as np
from dataprocess.processor import Processor

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex

def mergeSelect(unit):
    top20 = tfidfrank.search(','.join(unit.context[:-1]))
    unitset = UnitSet()
    for top in top20:
        tmp = Unit()
        tmp.context = top
        unitset.allunit.append(tmp)
    processor = Processor()

    # import ipdb
    # ipdb.set_trace()
    unitset = processor.run(unitset=unitset)
    hredranks = softmax([hredrank.distance(unit, t) for t in unitset])
    # hredrank.score(t,unitset)
    baseranks = softmax([baserank.distance(unit, t) for t in unitset])
    encoderanks = softmax([encoderank.distance(unit, t) for t in unitset])
    hredencoderanks = softmax([hredencoderank.distance(unit, t) for t in unitset])
    mergeranks = [sum(ranks) for ranks in zip(hredranks,baseranks,encoderanks,hredencoderanks)]
    sorted_merged_ranks = sorted(list(enumerate(mergeranks)),key=lambda item: item[1],reverse=True)
    top_index = sorted_merged_ranks[1][0]
    return top20[top_index]


if __name__ == '__main__':
    try:
        input_file_path = sys.argv[1]
        output_file_path = sys.argv[2]
    except IndexError:
        print('Two positional arguments are required: input_file_path, output_file_path')
        exit(1)

    if not os.path.isfile(input_file_path):
        print('Input file is not found: %s' % input_file_path)
        exit(1)

    output_dir = os.path.dirname(output_file_path)
    if not os.path.isdir(output_dir):
        print('Output dir is not exist: %s' % output_dir)
        exit(1)

    baserank = BaseRank()
    baserank.set('../data/allchat.set')

    hredrank = HredRank()
    hredrank.set('../data/allchat.set')

    file_obj = FileObj(r"../tfidf/tfidf_data/context.txt")
    train_sentences = file_obj.read_lines()[:100]
    tfidfrank = TfidfRank(train_sentences)
    tfidfrank.set('../data/allchat.set')

    hredencoderank = HredEncodeRank()
    hredencoderank.set('../data/allchat.set')
    encoderank = EncodeRank()
    encoderank.set('../data/allchat.set')

    file_obj = FileObj(input_file_path)
    test_sentences = file_obj.read_lines()
    file_result = open(output_file_path, 'w')
    for sentence in test_sentences:
        unit = Unit()
        unit.context = sentence.split('<s>')
        result = mergeSelect(unit)
        file_result.write(result)

