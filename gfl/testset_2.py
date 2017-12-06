#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Joseph Lefevre
"""

import scipy.io.wavfile as wav
import numpy as np
# with this example, we have n = 661500, p = 2 and k = 2
# k1 ~= 220500 and k2 ~= 441000

rate, signal = wav.read("../data/signal_audio_1.wav")
# signal is a ndarray with n = 661500 and p = 2
