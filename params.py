#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:08:06 2018

@author: juho
"""

import numpy as np
import itertools
import scipy.special
import sys


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


try:
    ranks_5cards = np.load('/home/juho/dev_folder/hand_eval/LUT_ranks_5cards.npy')
    ranks_7cards = np.load('/home/juho/dev_folder/hand_eval/LUT_ranks_7cards.npy')
except FileNotFoundError as err:
    print('VIRHE! Cannot load lookup tables for hand evaluator. Run create_5card_LUT.py and \
          create_7card_LUT.py before proceeding.\n')
    print(err)
    sys.exit()


combs = np.array(list(itertools.combinations(np.arange(7), 5))).flatten()     # Five card combinations out of seven cards
combs_numba = np.array(list(itertools.combinations(np.arange(7), 5)))

cardToInt, intToCard = getCardConversions()

LUT_nChooseK_5cards = np.zeros((52,5), dtype=np.int64)
LUT_nChooseK_5cards[:,0] = scipy.special.comb(np.arange(52), np.full(52,1)).astype(np.int)
LUT_nChooseK_5cards[:,1] = scipy.special.comb(np.arange(52), np.full(52,2)).astype(np.int)
LUT_nChooseK_5cards[:,2] = scipy.special.comb(np.arange(52), np.full(52,3)).astype(np.int)
LUT_nChooseK_5cards[:,3] = scipy.special.comb(np.arange(52), np.full(52,4)).astype(np.int)
LUT_nChooseK_5cards[:,4] = scipy.special.comb(np.arange(52), np.full(52,5)).astype(np.int)

LUT_nChooseK_7cards = np.zeros((52,7), dtype=np.int64)
LUT_nChooseK_7cards[:,0] = scipy.special.comb(np.arange(52), np.full(52,1)).astype(np.int)
LUT_nChooseK_7cards[:,1] = scipy.special.comb(np.arange(52), np.full(52,2)).astype(np.int)
LUT_nChooseK_7cards[:,2] = scipy.special.comb(np.arange(52), np.full(52,3)).astype(np.int)
LUT_nChooseK_7cards[:,3] = scipy.special.comb(np.arange(52), np.full(52,4)).astype(np.int)
LUT_nChooseK_7cards[:,4] = scipy.special.comb(np.arange(52), np.full(52,5)).astype(np.int)
LUT_nChooseK_7cards[:,5] = scipy.special.comb(np.arange(52), np.full(52,6)).astype(np.int)
LUT_nChooseK_7cards[:,6] = scipy.special.comb(np.arange(52), np.full(52,7)).astype(np.int)






