#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 18:05:29 2022

@author: mani
"""



import librosa
import numpy as np
import math

import cv2
from pathlib import Path
import sys
import os
import wave
import wavio as wv
import soundfile as sf
sys.path.insert(0, './features')
from flask import Flask, render_template, request

import pickle
import features.mlp_features as mlp_features
import features.cnn_features as cnn_features
import features.lstm_features as lstm_features

import audio_text.transformer_model as toText

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
app = Flask(__name__)

freq = 44100
duration = 8

global m
os.chdir(os.path.dirname(os.path.abspath(__file__)))
@app.route('/')
def title():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
   
            
        f = request.files['audio_data']
        
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
         

        print('file uploaded successfully')
        r_data, fs = sf.read('audio.wav')
        wv.write("test.wav", r_data, freq, sampwidth=2)
       
        return render_template('index.html', request="POST")
    else:
        return render_template("index.html")

@app.route('/model', methods=['GET', 'POST'])
def model():
    res = ['No Wake Word', 'Yes! Correct - Hi!']
    if request.method == "POST":
        models = request.form.get("models", None)


        if models == 'MLP':
            
            data = mlp_features.features_f('test.wav')
          
            sname = './trained_models/scaler.sav'
            sc = pickle.load(open(sname, 'rb'))
  
            
       
            data = sc.transform(data)
  
      
            fname = './trained_models/MLP.sav'
            loaded_model = pickle.load(open(fname, 'rb'))
            result = loaded_model.predict(data)
            print(result, 'tt')
            r1 = res[int(result)]
            transcription = toText.convert_text()
            return render_template("index.html", r1=r1, transcription = transcription, models=models)
        
        elif models == 'CNN':
            cnn_features.get_images('test.wav')
            img = cv2.imread('test.png')
          
            img = img.astype(np.float32) / 255.0
            img = (img - 0.5) * 2
            cnn_data = cv2.resize(img, (64, 64))
            
            cnn_data =np.expand_dims(cnn_data, axis=0)
          
        
    
            c_model_saved = load_model('./trained_models/spotify_CNN_keras_history.h5')
            y_pred = c_model_saved.predict(cnn_data/500)
    
            y_pred = np.argmax(y_pred, axis=1)
     
            r1 = res[int(y_pred)] 
            transcription = toText.convert_text()
      
            return render_template("index.html", r1 = r1, transcription = transcription, models=models)
        
        elif models == 'LSTM':
            d = lstm_features.features_f('test.wav')
         
            l_sname = './trained_models/LSTM_scaler.sav'
            l_sc = pickle.load(open(l_sname, 'rb'))
            d = l_sc.transform(d.reshape(d.shape[0], -1)).reshape(d.shape)
            d = np.array(d)
            model_saved = load_model('./trained_models/LSTM_model_history.h5')
            y_pred = model_saved.predict(d)
            y_pred = (y_pred > 0.5).astype('int')
            r1 = res[int(y_pred)]
            transcription = toText.convert_text()
            return render_template("index.html", r1 = r1, transcription = transcription, models = models)
   
      
        if models!=None:
         
            return render_template("index.html", models = models)
         
    return render_template("index.html", models = models)





if __name__ == '__main__':
   app.debug = True
   port = int(os.environ.get('PORT', 5000))
   app.run(debug=True, host='0.0.0.0', port=port)