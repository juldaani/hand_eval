#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:10:48 2018

@author: juho
"""


import time
import numpy as np
import scipy.special

n = 52
k = 5
ranks = np.load('LUT_ranks.npy')

st = time.time()

asd = np.random.choice(52, size=5, replace=0)
asd = np.tile(asd, (4000000,1))
asd = np.sort(asd,1)

fact = np.tile(np.arange(1, k+1), (len(asd), 1))
inds = np.sum(scipy.special.comb(asd, fact), axis=1).astype(np.int)

rank = ranks[inds].astype(np.int)

print(time.time() - st)


