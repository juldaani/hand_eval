#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 18:13:41 2019

@author: juho
"""


import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def evaluate_numba(cards, combs, ranks, LUT_nChooseK):
    if(len(cards) < 7): return None
    
    # Sorting the cards is mandatory for the algorithm to function properly
    cards = np.sort(cards)
    
    # 5 card combinations out of 7 cards
    fiveCardHands = cards[combs].reshape((-1,5))
   
    # Compute combinatorial indexes
    # https://en.wikipedia.org/wiki/Combinatorial_number_system
    tmp = np.zeros((21,5), dtype=np.int64)
    for i in range(21):
        tmp[i,0] = LUT_nChooseK[fiveCardHands[i,0],0]
        tmp[i,1] = LUT_nChooseK[fiveCardHands[i,1],1]
        tmp[i,2] = LUT_nChooseK[fiveCardHands[i,2],2]
        tmp[i,3] = LUT_nChooseK[fiveCardHands[i,3],3]
        tmp[i,4] = LUT_nChooseK[fiveCardHands[i,4],4]
    inds = np.zeros(21, dtype=np.int64)
    for i in range(21):
        inds[i] = np.sum(tmp[i,:])
    
    cardRanks = ranks[inds]
    bestIdx = np.argmin(cardRanks)
    bestHand = fiveCardHands[bestIdx]
    rank = cardRanks[bestIdx]

    return rank, bestHand


















