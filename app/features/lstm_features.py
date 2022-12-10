#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 20:31:21 2022

@author: mani
"""

import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

n_mfccs = 128

max_len= 300
def features_f(test):
    features=[] #list to save features
    labels=[] #list to save labels

    
    y , sr = librosa.load(test)
 
    
    mfccs = librosa.feature.mfcc(y, sr = sr, n_mfcc=n_mfccs, hop_length = max_len)
      # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfccs.shape[1]):
        pad_width = max_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

      # Else cutoff the remaining parts
    else:
        mfccs = mfccs[:, :max_len]
    mfccs = np.array(mfccs)
    features.append(mfccs)
   
 
    output=np.concatenate(features,axis=0)


    return(np.array(features))
