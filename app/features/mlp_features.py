#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 16:50:19 2022

@author: mani
"""
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def features_f(test):


    feature_dict = {'rmse':[],'chroma_stft':[], 'spec_cent':[],'spec_bw':[],'rolloff':[],'zcr':[],'mfcc':[], 
                'rmse_min':[],'chroma_stft_min':[], 'spec_cent_min':[],'spec_bw_min':[],'rolloff_min':[],'zcr_min':[],
                'mfcc_min':[], 'rmse_max':[],'chroma_stft_max':[], 'spec_cent_max':[],'spec_bw_max':[],'rolloff_max':[],'zcr_max':[],
                'mfcc_max':[]}


     
  
        
    x , sr = librosa.load(test)



    rmse = librosa.feature.rms(y=x)
    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=x, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(x)
    mfcc = librosa.feature.mfcc(y=x, sr=sr)
        

       
    # mean
    
    feature_dict['rmse'].append(np.mean(rmse))
    feature_dict['chroma_stft'].append(np.mean(chroma_stft))
    feature_dict['spec_cent'].append(np.mean(spec_cent))
    feature_dict['spec_bw'].append(np.mean(spec_bw))
    feature_dict['rolloff'].append(np.mean(rolloff))
    feature_dict['zcr'].append(np.mean(zcr))
    feature_dict['mfcc'].append(np.mean(mfcc))
        
 
    # min
    feature_dict['rmse_min'].append(np.min(rmse))
    feature_dict['chroma_stft_min'].append(np.min(chroma_stft))
    feature_dict['spec_cent_min'].append(np.min(spec_cent))
    feature_dict['spec_bw_min'].append(np.min(spec_bw))
    feature_dict['rolloff_min'].append(np.min(rolloff))
    feature_dict['zcr_min'].append(np.min(zcr))
    feature_dict['mfcc_min'].append(np.min(mfcc))
   
    # max
    feature_dict['rmse_max'].append(np.max(rmse))
    feature_dict['chroma_stft_max'].append(np.max(chroma_stft))
    feature_dict['spec_cent_max'].append(np.max(spec_cent))
    feature_dict['spec_bw_max'].append(np.max(spec_bw))
    feature_dict['rolloff_max'].append(np.max(rolloff))
    feature_dict['zcr_max'].append(np.max(zcr))
    feature_dict['mfcc_max'].append(np.max(mfcc))
        
    df_test = pd.DataFrame(feature_dict)

    X_test = np.array(df_test)



    return X_test