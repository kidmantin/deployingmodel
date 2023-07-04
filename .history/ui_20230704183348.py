import streamlit as st

st.title('AYO CHILL')
# st.file_uploader()
button1 = st.button('chill?')

if button1:
    st.write('chilled')
    
like = st.chechbox('why tho?')

button2 = st.button("let's see")

if button2:
    if like:
        st.write('asd')
    else: st.write('nosdsd')