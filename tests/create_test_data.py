#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 19:00:25 2018

@author: juho

Create test data for hand evaluator unit tests. Deuces hand evaluator 
(https://github.com/worldveil/deuces) and Python 2 is required to run this 
script.

"""


from deuces import Card
from deuces import Evaluator
import itertools
import numpy as np
import scipy.special
from hand_eval.params import intToCard, cardToInt

n = 52
k = 5
combs = np.array(list(itertools.combinations(np.arange(n), k)))     # all possible 5 card combinations

evaluator = Evaluator()

ranks = []
cards = []
for i,comb in enumerate(combs):
    if(i%10000==0): print(i)
    cardsText = [intToCard[comb[0]], intToCard[comb[1]], intToCard[comb[2]],
             intToCard[comb[3]], intToCard[comb[4]]]

    board = [Card.new(intToCard[comb[0]]),
            Card.new(intToCard[comb[1]]),
            Card.new(intToCard[comb[2]])]
    hand = [Card.new(intToCard[comb[3]]),
            Card.new(intToCard[comb[4]])]
    
    rank = evaluator.evaluate(board, hand)
    
    ranks.append(rank)
    cards.append(cardsText)
    
np.save('cardsTextFormat', np.array(cards))
np.save('ranks', np.array(ranks))
np.save('cardsIntFormat', combs)
    
    
    