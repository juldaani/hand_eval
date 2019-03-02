#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 18:13:41 2019

@author: juho

Evaluators for 7 card hands. See 'tests/unit_tests.py' how to use them.

"""


import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def sortCards(cards):
    """
    Sort 7 card hand using sorting network.
    """
    
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
    
    return cards


@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def evaluate_numba2(cards, ranks_7cards, LUT_nChooseK_7cards, tmp):
    """
    Evaluate 7 card hands with direct lookup. Ranks for all 7 card combinations 
    are stored in one massive lookup table where lower rank means better hand. 
    Ranks follow the same scheme as in the Deuces evaluator: 
    https://github.com/worldveil/deuces. Card combinations in the lookup table 
    are indexed using combinatorial number system:
    https://en.wikipedia.org/wiki/Combinatorial_number_system.
    
    cards -- Array of cards encoded as integers between 0-51, see 'params.py'.
    ranks_7cards - Lookup table for all 7 card combinations, this can be 
        generated with 'create_7card_LUT.py' or downloaded from:
        https://drive.google.com/open?id=1sNcEmyEAqkMrR8fDS0tSkr2KFcgx8gKj
    LUT_nChooseK_7cards - Lookup table for computing index for the hand, see 
        'params.py'.
    tmp -- Should be a following array: np.zeros(7, dtype=np.int64)
           This is to prevent memory allocation if the evaluator is used inside
           the loop.
    """
    
    if(len(cards) < 7): return None
    
    cards = sortCards(cards)

    tmp[0] = LUT_nChooseK_7cards[cards[0],0]
    tmp[1] = LUT_nChooseK_7cards[cards[1],1]
    tmp[2] = LUT_nChooseK_7cards[cards[2],2]
    tmp[3] = LUT_nChooseK_7cards[cards[3],3]
    tmp[4] = LUT_nChooseK_7cards[cards[4],4]
    tmp[5] = LUT_nChooseK_7cards[cards[5],5]
    tmp[6] = LUT_nChooseK_7cards[cards[6],6]
    ind = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6]

    return ranks_7cards[ind]



@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def evaluate_numba(cards, combs, ranks_5cards, LUT_nChooseK_5cards):
    """
    Evaluate 7 card hand by decomposing the hand first into all possible 5 card 
    combinations and then executing direct lookup for each of the 5 card 
    combinations. Although this method is slower than 'evaluate_numba2', it 
    can provide the winning 5 card hand in addition to hand rank.
    
    cards -- Array of cards encoded as integers between 0-51, see 'params.py'.
    combs - Indexes for all 5 card combinations out of 7 card hand, see 'params.py'
    ranks_5cards - Lookup table for all 5 card combinations, this can be 
        generated with 'creare_5card_LUT.py' or downloaded from:
        https://drive.google.com/open?id=1sNcEmyEAqkMrR8fDS0tSkr2KFcgx8gKj
    LUT_nChooseK_5cards - Lookup table for computing index for the hand, see 
        'params.py'.
    """
    
    if(len(cards) < 7): return None
    
    # Sorting the cards is mandatory for the algorithm to function properly
    cards = sortCards(cards)
    
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








