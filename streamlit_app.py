import streamlit as st
from audio_recorder_streamlit import audio_recorder
import io
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import training
import plotting_functions
import requests

SAMPLE_RATE = 16000

labels = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes'] 

st.title('Label identifier')

st.write(f'possible labels: {labels}')

# for microphone input
audio_data = audio_recorder(text='Microphone input, audio will be recorded for 1 second', energy_threshold=(-1.0, 1.0),
                            pause_threshold=1, sample_rate=SAMPLE_RATE)

if audio_data is not None:
    voice_data_2channels, samplerate = sf.read(io.BytesIO(audio_data), frames=16000)
    voice_data_mono = voice_data_2channels[:, 0]
    st.audio(voice_data_mono, sample_rate=samplerate)
    
    if st.button('plots'):
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(voice_data_mono.shape[0])
        axes[0].plot(timescale, voice_data_mono)
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])
        
        plotting_functions.plot_spectrogram(training.get_spectrogram(voice_data_mono), axes[1])
        axes[1].set_title('Spectrogram')
        
        st.pyplot(fig)
        
    if st.button('predict'):
        
        response = requests.post( 
        # "http://127.0.0.1:3000/classify",
        "https://audio-commands-787b5415843a.herokuapp.com/classify",
        headers={"content-type": "application/json"},
        json=voice_data_mono.tolist(),
        ).json()
        
        fig, ax = plt.subplots()
        ax.bar(labels, tf.nn.softmax(response[0]))
        ax.set_title(f'Predicted probs, predicted label: {tf.argmax(tf.nn.softmax(response[0]))}')
        
        st.pyplot(fig)