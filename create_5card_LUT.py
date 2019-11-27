#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 15:40:00 2019

@author: juho

Create 5 card lookup table to be used with hand evaluators. To run this script, 
Deuces hand evaluator (https://github.com/worldveil/deuces) and Python 2 are 
required.

Lookup tables can also be downloaded from: https://drive.google.com/drive/folders/1sNcEmyEAqkMrR8fDS0tSkr2KFcgx8gKj
"""

from deuces import Evaluator, Card

import numpy as np
import scipy.special
import itertools

from hand_eval.params import cardToInt, intToCard



n = 52
k = 5
combs = np.array(list(itertools.combinations(np.arange(n), k)))
fact = np.tile(np.arange(1, k+1), (len(combs), 1))
inds = np.sum(scipy.special.comb(combs, fact), axis=1).astype(np.int)

evaluator = Evaluator()
ranks = np.zeros(len(combs), np.uint16)
for i in range(len(combs)):
    comb = combs[i]
    idx = inds[i]

    board = [Card.new(intToCard[comb[0]]),
            Card.new(intToCard[comb[1]]),
            Card.new(intToCard[comb[2]])]
    hand = [Card.new(intToCard[comb[3]]),
            Card.new(intToCard[comb[4]])]

    rank = evaluator.evaluate(board, hand)

    ranks[idx] = rank

    if(i % 100000 == 0): print(i, len(combs))

np.save('LUT_ranks_5cards.npy', ranks)
















