#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:08:48 2018

@author: juho
"""

import numpy as np
import scipy.special
from hand_eval.params import combs, ranks



def evaluate(cards):
    # Check that all cards are unique
    if(len(np.unique(cards)) < 7):
        return None
    
#    cards = np.random.choice(52, size=7, replace=0)
    cards = np.sort(np.array(cards))
    fiveCardHands = cards[combs].reshape((-1,5))
    
    fact = np.tile(np.arange(1, 5+1), (len(fiveCardHands), 1))
    inds = np.sum(scipy.special.comb(fiveCardHands, fact), axis=1).astype(np.int)
    
    cardRanks = ranks[inds].astype(np.int)
    
    bestIdx = np.argmin(cardRanks)
    bestHand = fiveCardHands[bestIdx]
    rank = cardRanks[bestIdx]

    return rank, bestHand


