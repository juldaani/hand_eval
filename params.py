#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:08:06 2018

@author: juho
"""

import numpy as np
import itertools

def getCardConversions():
    cardNums = 'A 2 3 4 5 6 7 8 9 T J Q K'
    cardNums = cardNums.split()
    cardSuits = 'c d h s'
    cardSuits = cardSuits.split()

    cardToInt = {}
    intToCard = {}
    
    c = 0
    for suit in cardSuits:
        for num in cardNums:
            cardToInt[num+suit] = c
            intToCard[c] = num+suit
            c += 1

    return cardToInt, intToCard

ranks = np.load('/home/juho/dev_folder/hand_eval/LUT_ranks.npy')
combs = np.array(list(itertools.combinations(np.arange(7), 5))).flatten()     # Five card hands out of seven cards
cardToInt, intToCard = getCardConversions()