#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Joseph Lefevre
"""

import scipy.signal

n1 = 30
n2 = 50
n3 = 20
p = 1
k = 2

std = 10

signal1 = scipy.signal.gaussian(n1, 10)
signal2 = scipy.signal.gaussian(n1, 10) + 10
signal3 = scipy.signal.gaussian(n1, 10) - 10

signal = np.concatenate([signal1, signal2, signal3])

# the true breaking points are 30 and 80

