import streamlit as st
import training
import io
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import bentoml

st.title('AYO CHILL')
file = st.file_uploader('audio of command')
button1 = st.button('chill?')

if button1:
    st.write(file)
    st.audio(file)

def preprocess_file(waveform):
    spectrogram = training.get_spectrogram(waveform)
    # output_ds = output_ds.map(
    #     map_func=get_spectrogram_and_label_id,
    #     num_parallel_calls=AUTOTUNE)
    # return output_ds

if st.button('check'):
    # st.write(type(file.read()))
    # data, samplerate = sf.read(io.BytesIO(file.read()))
    data, samplerate = sf.read(io.BytesIO(file.read()))
    st.audio(data, sample_rate=samplerate)
    
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(data.shape[0])
    axes[0].plot(timescale, data)
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    
    st.pyplot(fig)
    # preprocess_file(data)
    # processed = training.preprocess_dataset(data)
    # st.write(type(processed))

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