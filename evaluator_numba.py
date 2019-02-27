#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 18:13:41 2019

@author: juho
"""


import numpy as np
from numba import jit



@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def evaluate_numba2(cards, ranks_7cards, LUT_nChooseK_7cards, tmp):
    """
    Evaluate 7 card hand with direct lookup.
    
    tmp -- Should be a following array: np.zeros(7, dtype=np.int64)
           This is to prevent memory allocation if the evaluator is used inside
           the loop.
    """
    
    if(len(cards) < 7): return None
    
    if(cards[1] > cards[2]): cards[1], cards[2] = cards[2], cards[1]
    if(cards[3] > cards[4]): cards[3], cards[4] = cards[4], cards[3]
    if(cards[5] > cards[6]): cards[5], cards[6] = cards[6], cards[5]
    
    if(cards[0] > cards[2]): cards[0], cards[2] = cards[2], cards[0]
    if(cards[3] > cards[5]): cards[3], cards[5] = cards[5], cards[3]
    if(cards[4] > cards[6]): cards[4], cards[6] = cards[6], cards[4]

    if(cards[0] > cards[1]): cards[0], cards[1] = cards[1], cards[0]
    if(cards[4] > cards[5]): cards[4], cards[5] = cards[5], cards[4]
    if(cards[2] > cards[6]): cards[2], cards[6] = cards[6], cards[2]

    if(cards[0] > cards[4]): cards[0], cards[4] = cards[4], cards[0]
    if(cards[1] > cards[5]): cards[1], cards[5] = cards[5], cards[1]
    
    if(cards[0] > cards[3]): cards[0], cards[3] = cards[3], cards[0]
    if(cards[2] > cards[5]): cards[2], cards[5] = cards[5], cards[2]

    if(cards[1] > cards[3]): cards[1], cards[3] = cards[3], cards[1]
    if(cards[2] > cards[4]): cards[2], cards[4] = cards[4], cards[2]

    if(cards[2] > cards[3]): cards[2], cards[3] = cards[3], cards[2]

    tmp[0] = LUT_nChooseK_7cards[cards[0],0]
    tmp[1] = LUT_nChooseK_7cards[cards[1],1]
    tmp[2] = LUT_nChooseK_7cards[cards[2],2]
    tmp[3] = LUT_nChooseK_7cards[cards[3],3]
    tmp[4] = LUT_nChooseK_7cards[cards[4],4]
    tmp[5] = LUT_nChooseK_7cards[cards[5],5]
    tmp[6] = LUT_nChooseK_7cards[cards[6],6]
    ind = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]

    return ranks_7cards[ind]



#@jit(nopython=True, cache=True, fastmath=True, nogil=True)
#def evaluate_numba2(cards, ranks_7cards, LUT_nChooseK_7cards):
#    """
#    Evaluate 7 card hand with direct lookup.
#    """
#    
#    if(len(cards) < 7): return None
#    
#    # Sorting the cards is mandatory for the algorithm to function properly
#    cards = np.sort(cards)
#    
#    tmp = np.zeros(7, dtype=np.int64)
#    tmp[0] = LUT_nChooseK_7cards[cards[0],0]
#    tmp[1] = LUT_nChooseK_7cards[cards[1],1]
#    tmp[2] = LUT_nChooseK_7cards[cards[2],2]
#    tmp[3] = LUT_nChooseK_7cards[cards[3],3]
#    tmp[4] = LUT_nChooseK_7cards[cards[4],4]
#    tmp[5] = LUT_nChooseK_7cards[cards[5],5]
#    tmp[6] = LUT_nChooseK_7cards[cards[6],6]
#    ind = np.sum(tmp)
#
#    rank = ranks_7cards[ind]
#   
#    return rank


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def evaluate_numba(cards, combs, ranks_5cards, LUT_nChooseK_5cards):
    """
    Evaluate 7 card hand by decomposing it into 5 card combinations.
    """
    
    if(len(cards) < 7): return None
    
    # Sorting the cards is mandatory for the algorithm to function properly
    cards = np.sort(cards)
    
    # 5 card combinations out of 7 cards
    fiveCardHands = np.zeros((21,5), dtype=np.int64)
    for i in range(len(combs)):
        fiveCardHands[i,0] = cards[combs[i,0]]
        fiveCardHands[i,1] = cards[combs[i,1]]
        fiveCardHands[i,2] = cards[combs[i,2]]
        fiveCardHands[i,3] = cards[combs[i,3]]
        fiveCardHands[i,4] = cards[combs[i,4]]
   
    # Compute combinatorial indexes
    # https://en.wikipedia.org/wiki/Combinatorial_number_system
    tmp = np.zeros((21,5), dtype=np.int64)
    for i in range(21):
        tmp[i,0] = LUT_nChooseK_5cards[fiveCardHands[i,0],0]
        tmp[i,1] = LUT_nChooseK_5cards[fiveCardHands[i,1],1]
        tmp[i,2] = LUT_nChooseK_5cards[fiveCardHands[i,2],2]
        tmp[i,3] = LUT_nChooseK_5cards[fiveCardHands[i,3],3]
        tmp[i,4] = LUT_nChooseK_5cards[fiveCardHands[i,4],4]
    inds = np.zeros(21, dtype=np.int64)
    for i in range(21):
        inds[i] = np.sum(tmp[i,:])
    
    cardRanks = ranks_5cards[inds]
    bestIdx = np.argmin(cardRanks)
    bestHand = fiveCardHands[bestIdx]
    rank = cardRanks[bestIdx]

    return rank, bestHand








