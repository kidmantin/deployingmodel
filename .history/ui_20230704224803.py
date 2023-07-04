import streamlit as st
import training
import io
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import bentoml

def preprocess_file(waveform):
    spectrogram = training.get_spectrogram(waveform)
    # output_ds = output_ds.map(
    #     map_func=get_spectrogram_and_label_id,
    #     num_parallel_calls=AUTOTUNE)
    return spectrogram

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
file = st.file_uploader('audio of command')
button1 = st.button('chill?')

if button1:
    st.write(file)
    st.audio(file)

data, samplerate = sf.read(io.BytesIO(file.read()))
st.audio(data, sample_rate=samplerate)

spectrogram = preprocess_file(data)

st.write(type(spectrogram))

if st.button('check'):
    # st.write(type(file.read()))
    # data, samplerate = sf.read(io.BytesIO(file.read()))
    
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(data.shape[0])
    axes[0].plot(timescale, data)
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    
    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    
    st.pyplot(fig)
    


st.download_button('download spectrogram', spectrogram, file_name='test_spectr')

# st.header("NO CHILL")
# like = st.checkbox('why tho?')

# button2 = st.button("let's see")

# if button2:
#     if like:
#         st.write('asd')
#     else: st.write('nosdsd')
    
# animals = st.radio("wahtr animal?", ["tuple", "list"])

# bimb = st.multiselect("wahtr animal?", ("bam", "bus"))


# button3 = st.button("let us see")

# if button3:
#     st.write(type(bimb))
    
# user_text = st.text_input("aworhgasf", "no bim")

# if st.button("bim?"):
#     st.write(user_text)
    
# st.write('sentiment', kaif())

# st.text_area('', "sdfosydf9p8u")