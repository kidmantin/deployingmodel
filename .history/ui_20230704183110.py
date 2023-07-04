import streamlit as st

st.title('AYO CHILL')
# st.file_uploader()
button1 = st.button('chill?')

if button1:
    st.write('chilled')