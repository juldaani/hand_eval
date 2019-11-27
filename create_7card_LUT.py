#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 11:14:36 2019

@author: juho

Create 7 card lookup table to be used with hand evaluator. To run this script, 
the 'create_5cards_LUT.py' must be executed first because this script uses the 
output from 'create_5cards_LUT.py'.

Lookup tables can also be downloaded from: https://drive.google.com/drive/folders/1sNcEmyEAqkMrR8fDS0tSkr2KFcgx8gKj

"""


import numpy as np
from hand_eval.evaluator_numba import evaluate_numba
from hand_eval.params import combs_numba, ranks_5cards, LUT_nChooseK_5cards, LUT_nChooseK_7cards
from numba import jit



@jit(nopython=True, cache=True, fastmath=True, nogil=True)
def createRanks_7cards(ranks_5cards, combs_numba, LUT_nChooseK_5cards, LUT_nChooseK_7cards):
    cards = np.arange(52)
    ranks_7cards = np.zeros(133784560, dtype=np.uint16)
    
    counter = 0
    tmpCards = np.zeros(7, dtype=np.int8)
    for i1 in range(52):
        c1 = cards[i1]
        
        for i2 in range(i1+1,52):
            c2 = cards[i2]
        
            for i3 in range(i2+1,52):
                c3 = cards[i3]
                
                for i4 in range(i3+1,52):
                    c4 = cards[i4]
                    
                    for i5 in range(i4+1,52):
                        c5 = cards[i5]
                        
                        for i6 in range(i5+1,52):
                            c6 = cards[i6]
                            
                            for i7 in range(i6+1,52):
                                c7 = cards[i7]
                                
                                tmpCards[0] = c1
                                tmpCards[1] = c2
                                tmpCards[2] = c3
                                tmpCards[3] = c4
                                tmpCards[4] = c5
                                tmpCards[5] = c6
                                tmpCards[6] = c7
                                
                                rank, _ = evaluate_numba(tmpCards, combs_numba, ranks_5cards, 
                                                         LUT_nChooseK_5cards)

                                tmp = np.zeros(7, dtype=np.uint64)
                                tmp[0] = LUT_nChooseK_7cards[c1,0]
                                tmp[1] = LUT_nChooseK_7cards[c2,1]
                                tmp[2] = LUT_nChooseK_7cards[c3,2]
                                tmp[3] = LUT_nChooseK_7cards[c4,3]
                                tmp[4] = LUT_nChooseK_7cards[c5,4]
                                tmp[5] = LUT_nChooseK_7cards[c6,5]
                                tmp[6] = LUT_nChooseK_7cards[c7,6]
                                ind = np.sum(tmp)

                                ranks_7cards[ind] = rank
                                
                                counter += 1
                                
                                if(counter % 1000000 == 0): print(counter/1000000)
    
    return ranks_7cards

                
ranks_7cards = createRanks_7cards(ranks_5cards, combs_numba, LUT_nChooseK_5cards, 
                                  LUT_nChooseK_7cards)

np.save('LUT_ranks_7cards', ranks_7cards)
