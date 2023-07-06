import streamlit as st
import training
import io
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import pandas as pd 
import training

labels = open(r'C:\Users\Kidma\source\repos\deployingmodel\labels.txt').read().split()

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

st.title('AYO CHILL')
uploaded_file = st.file_uploader('audio of command')
st.write(f'possible labels:')

if uploaded_file is not None:
    data, samplerate = sf.read(io.BytesIO(uploaded_file.read()))
    st.audio(data, sample_rate=samplerate)


    if st.button('plots'):
        fig, axes = plt.subplots(2, figsize=(12, 8))
        timescale = np.arange(data.shape[0])
        axes[0].plot(timescale, data)
        axes[0].set_title('Waveform')
        axes[0].set_xlim([0, 16000])
        
        
        plot_spectrogram(training.get_spectrogram(data), axes[1])
        axes[1].set_title('Spectrogram')
        
        st.pyplot(fig)

    model_keras = tf.keras.models.load_model(r'C:\Users\Kidma\source\repos\deployingmodel\audio_commands_model_keras')

    if st.button('predict'):
        input_data = training.preprocess_file(data)
        output = model_keras(input)
        
        fig, ax = plt.subplots()
        ax.bar(labels, tf.nn.softmax(output[0]))
        ax.set_title(f'Predicted probs')
        
        st.pyplot(fig)