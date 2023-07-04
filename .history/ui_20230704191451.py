import streamlit as st
import training
import io
import soundfile as sf

# def kaif():
#     st.write("kaif received")
#     return True

st.title('AYO CHILL')
file = st.file_uploader('audio of command')
button1 = st.button('chill?')

if button1:
    st.write(file)
    st.audio(file)

if st.button('check'):
    data, samplerate = sf.read(io.BytesIO(file))
    st.write(data)
    st.write(samplerate)
    # processed = training.preprocess_dataset(file)
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