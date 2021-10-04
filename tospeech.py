# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:17:00 2021

@author: 91820
"""
#include following libraries in the main file from where this file will be imported
import gtts
from time import sleep
import os
import pyglet


def tts(text):
    #language code for american english
    language = 'en' 
    #top level domain to use for different accents. "com" for United states
    tld = 'com'
    # "gTTS" creates an object converting text to speech
    speech = gtts.gTTS(text, tld=tld, lang=language, slow=False)
    
    filename = 'temp.mp3'
    speech.save(filename)
    
    # plays the audio through system without media player
    music = pyglet.media.load(filename, streaming=False)
    music.play()
    
    sleep(music.duration) #prevent from killing
    os.remove(filename) #remove temperory file
    
# Test Case
tts("This is a test note, choosen language is english and accent is American.")