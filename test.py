# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from deuces import Card

card = Card.new('Qh')



import itertools
import numpy as np
import scipy.special


n = 52
k = 5
combs = np.array(list(itertools.combinations(np.arange(n), k)))

fact = np.tile(np.arange(1, k+1), (len(combs), 1))
inds = np.sum(scipy.special.comb(combs, fact), axis=1).astype(np.int)


# %%
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


# %%

from deuces import Evaluator

evaluator = Evaluator()

ranks = np.zeros(len(combs))
for i in range(len(combs)):
    comb = combs[i]
    idx = inds[i]

#    print(intToCard[comb[4]])

    board = [Card.new(intToCard[comb[0]]),
            Card.new(intToCard[comb[1]]),
            Card.new(intToCard[comb[2]])]
    hand = [Card.new(intToCard[comb[3]]),
            Card.new(intToCard[comb[4]])]
    
    rank = evaluator.evaluate(board, hand)

    ranks[idx] = rank
    
    if(i % 10000 == 0): print(i)
    
#    print(rank)
#    print(board)
#    print(hand)
#    print(' ')

# %%


deucesRanks = []
ownRanks = []
#for i in range(10000):
for i,c in enumerate(combs):
#    c = np.sort(np.random.choice(52, size=5, replace=0))
    c = np.tile(c, (2,1))
    
    if(i % 10000 == 0): print(i)
    
    fact = np.tile(np.arange(1, k+1), (len(c), 1))
    inds = np.sum(scipy.special.comb(c, fact), axis=1).astype(np.int)
    
    rank = ranks[inds[0]].astype(np.int)
    ownRanks.append(rank)
    
    board = [Card.new(intToCard[c[0,0]]),
            Card.new(intToCard[c[0,1]]),
            Card.new(intToCard[c[0,2]])]
    hand = [Card.new(intToCard[c[0,3]]),
            Card.new(intToCard[c[0,4]])]
    deucesRank = evaluator.evaluate(board, hand)
    deucesRanks.append(deucesRank)

deucesRanks = np.array(deucesRanks)
ownRanks = np.array(ownRanks)


print(np.sum(deucesRanks == ownRanks))

#Card.print_pretty_cards(board + hand)


# %%


import time

st = time.time()

asd = np.random.choice(52, size=5, replace=0)
asd = np.tile(asd, (4000000,1))
asd = np.sort(asd,1)

fact = np.tile(np.arange(1, k+1), (len(asd), 1))
inds = np.sum(scipy.special.comb(asd, fact), axis=1).astype(np.int)

rank = ranks[inds].astype(np.int)

print(time.time() - st)



















