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

st.title('AYO CHILL')
uploaded_file = st.file_uploader('audio of command')
st.write(f'possible labels: {labels}')

audio_bytes = audio_recorder(text='Microphone input', pause_threshold=0.1, sample_rate=SAMPLE_RATE)

if audio_bytes is not None:
    st.write(type(audio_bytes))

    voice_data, samplerate = sf.read(io.BytesIO(audio_bytes))
    # st.write(samplerate)
    st.audio(voice_data, sample_rate=samplerate)
    
    if st.button('plots'):
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(voice_data.shape[0])
        axes[0].plot(timescale, voice_data)
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])
        
        plotting_functions.plot_spectrogram(training.get_spectrogram(voice_data), axes[1])
        axes[1].set_title('Spectrogram')
        
        st.pyplot(fig)

    
    if st.button('predict'):
        
        response = requests.post( 
        # "http://127.0.0.1:3000/classify",
        "https://audio-commands-787b5415843a.herokuapp.com/classify",
        headers={"content-type": "application/json"},
        json=voice_data.tolist(),
        ).json()
        
        fig, ax = plt.subplots()
        ax.bar(labels, tf.nn.softmax(response[0]))
        ax.set_title(f'Predicted probs')
        
        st.pyplot(fig)

if uploaded_file is not None:
    uploaded_data, samplerate = sf.read(io.BytesIO(uploaded_file.read()))
    st.audio(uploaded_data, sample_rate=samplerate)
    
    if st.button('plots'):
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(uploaded_data.shape[0])
        axes[0].plot(timescale, uploaded_data)
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])
        
        plotting_functions.plot_spectrogram(training.get_spectrogram(uploaded_data), axes[1])
        axes[1].set_title('Spectrogram')
        
        st.pyplot(fig)

    if st.button('predict'):
        
        response = requests.post( 
        # "http://127.0.0.1:3000/classify",
        "https://audio-commands-787b5415843a.herokuapp.com/classify",
        headers={"content-type": "application/json"},
        json=uploaded_data.tolist(),
        ).json()
        
        fig, ax = plt.subplots()
        ax.bar(labels, tf.nn.softmax(response[0]))
        ax.set_title(f'Predicted probs')
        
        st.pyplot(fig)
        
        