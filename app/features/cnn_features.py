#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:14:44 2022

@author: mani


"""

import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.switch_backend('Agg') 

def get_images(test):

    print(test)
    y , sr = librosa.load(test)
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x = y, Fs= sr, noverlap=384, NFFT=512)
    ax.axis('off')
    fig.savefig('./test.png', dpi=300)
 

