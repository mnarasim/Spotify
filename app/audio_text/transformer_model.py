import librosa
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def convert_text():
    x , sr = librosa.load('test.wav', sr=16000)
    '''
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    input_values = tokenizer(x, return_tensors = "pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    transcription = tokenizer.batch_decode(prediction)[0]

    return transcription
    '''
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    input_values = processor(x, return_tensors = "pt").input_values
    logits = model(input_values).logits
    prediction = torch.argmax(logits, dim = -1)
    transcription = processor.batch_decode(prediction)[0]

    return transcription


