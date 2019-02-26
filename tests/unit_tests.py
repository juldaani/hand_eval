#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 19:21:17 2018

@author: juho
"""

import unittest
import numpy as np
from hand_eval.evaluator import evaluate
from hand_eval.evaluator_numba import evaluate_numba
from hand_eval.params import combs, ranks, LUT_nChooseK

class Tests(unittest.TestCase):
 
    def setUp(self):
        self.hands = np.load('cardsIntFormat.npy')
        self.ranks = np.load('ranks.npy')
        rankDict = {}
        for hand,rank in zip(self.hands,self.ranks):
            rankDict[tuple(hand)] = rank
        self.rankDict = rankDict
#        pass
 
    def testAreRanksCorrect(self):
        
        for i in range(1000000):
            if(i%10000==0): print(i)
            
            cards = np.sort(np.random.choice(52, size=7, replace=0))    # Random 7 cards
            fiveCardHands = cards[combs].reshape((-1,5))
            
            tmpRanks = []
            for hand in fiveCardHands:
                tmpRanks.append(self.rankDict[tuple(hand)])
            
            bestIdx = np.argmin(tmpRanks)
            trueRank = tmpRanks[bestIdx]
            trueBestHand = fiveCardHands[bestIdx]
            
            rank, bestHand = evaluate(cards)
            self.assertEqual(trueRank, rank)
            self.assertTrue(np.all(trueBestHand == bestHand))

            rank, bestHand = evaluate_numba(cards, combs, ranks, LUT_nChooseK)
            self.assertEqual(trueRank, rank)
            self.assertTrue(np.all(trueBestHand == bestHand))
            
    

if __name__ == '__main__':
    unittest.main()










