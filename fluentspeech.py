#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:12:02 2022

@author: umbertocappellazzo
"""

import os
import torch
from typing import Union
from torch.utils.data import Dataset
import numpy as np
from torch.nn import functional as F
from torchaudio import transforms as t
import soundfile

class FluentSpeech(Dataset):
    """FluentSpeechCommand dataset.
    Made of short audio with different speakers asking something.
    https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/
    """
    URL = "http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz"

    def __init__(self, data_path, max_len_audio, train: Union[bool, str] = True):
        if not isinstance(train, bool) and train not in ("train", "valid", "test"):
            raise ValueError(f"`train` arg ({train}) must be a bool or train/valid/test.")
            
        if isinstance(train, bool):
            if train:
                self.train = "train"
            else:
                self.train = "test"
        if train in ("train", "valid", "test"):
            self.train = train
        self.max_len_audio = max_len_audio
        self.data_path = os.path.expanduser(data_path)
        self.x, self.y, _, self.transcription = self.get_data()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def get_data(self):
        
        #map_transcripts = TextTransform() 
        base_path = os.path.join(self.data_path, "fluent_speech_commands_dataset")

        #self.transcriptions = []

        x, y, transcriptions = [], [], []

        with open(os.path.join(base_path, "data", f"{self.train}_data.csv")) as f:
            lines = f.readlines()[1:]

        for line in lines:
            items = line[:-1].split(',')
            

            action, obj, location = items[-3:]
            pathh = os.path.join(base_path, items[1])
            wav = soundfile.read(pathh)[0]

            if len(wav) > self.max_len_audio:
                pass
            else:

                x.append(os.path.join(base_path, items[1]))
                y.append(
                    self.class_ids[action+obj+location]    
                )
            
            #transcriptions.append(map_transcripts.text_to_int(items[3].lower()))
                transcriptions.append(items[3].lower())
            
        return np.array(x), np.array(y), None, transcriptions #np.array(transcriptions,dtype=object)
    
    
    @property
    def transformations(self):
        return None
    
    @property
    def class_ids(self):
        return {
             'change languagenonenone': 0,
             'activatemusicnone': 1,
             'activatelightsnone': 2,
             'deactivatelightsnone': 3,
             'increasevolumenone': 4,
             'decreasevolumenone': 5,
             'increaseheatnone': 6,
             'decreaseheatnone': 7,
             'deactivatemusicnone': 8,
             'activatelampnone': 9,
             'deactivatelampnone': 10,
             'activatelightskitchen': 11,
             'activatelightsbedroom': 12,
             'activatelightswashroom': 13,
             'deactivatelightskitchen': 14,
             'deactivatelightsbedroom': 15,
             'deactivatelightswashroom': 16,
             'increaseheatkitchen': 17,
             'increaseheatbedroom': 18,
             'increaseheatwashroom': 19,
             'decreaseheatkitchen': 20,
             'decreaseheatbedroom': 21,
             'decreaseheatwashroom': 22,
             'bringnewspapernone': 23,
             'bringjuicenone': 24,
             'bringsocksnone': 25,
             'change languageChinesenone': 26,
             'change languageKoreannone': 27,
             'change languageEnglishnone': 28,
             'change languageGermannone': 29,
             'bringshoesnone': 30
        }




def trunc(x, max_len):
    l = len(x)
    if l > max_len:
        x = x[l//2-max_len//2:l//2+max_len//2]
    if l < max_len:
        x = F.pad(x, (0, max_len-l), value=0.)
    
    eps = np.finfo(np.float64).eps
    sample_rate = 16000
    n_mels = 40
    win_len = 25
    hop_len= 10
    win_len = int(sample_rate/1000*win_len)
    hop_len = int(sample_rate/1000*hop_len)
    mel_spectr = t.MelSpectrogram(sample_rate=16000,
            win_length=win_len, hop_length=hop_len, n_mels=n_mels)
    
    return np.log(mel_spectr(x)+eps) 


if __name__ == '__main__':
    data_path ='/home/ste/Datasets/'
    a = FluentSpeech(data_path,train=True, max_len_audio=64000)
    # class_order = [19, 27, 30, 28, 15,  4,  2,  9, 10, 22, 11,  7,  1, 25, 16, 14,  5,
    #          8, 29, 12, 21, 17,  3, 20, 23,  6, 18, 24, 26,  0, 13]
    #loader = DataLoader(a, batch_size=4, shuffle=True, collate_fn=lambda x: data_processing(x, processor = processor))

    for i, (x,y) in enumerate(a):
        print(i, x, y)
    #print(type(a.get_data()))