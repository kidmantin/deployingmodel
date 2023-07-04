import streamlit as st

def kaif():
    st.write("kaif received")

st.title('AYO CHILL')
# st.file_uploader()
button1 = st.button('chill?')

if button1:
    st.write('chilled')

st.header("NO CHILL")
like = st.checkbox('why tho?')

button2 = st.button("let's see")

if button2:
    if like:
        st.write('asd')
    else: st.write('nosdsd')
    
animals = st.radio("wahtr animal?", ["tuple", "list"])

bimb = st.multiselect("wahtr animal?", ("bam", "bus"))


button3 = st.button("let us see")

if button3:
    st.write(type(bimb))
    
user_text = st.text_input("aworhgasf", "no bim")

if st.button("bim?"):
    st.write(user_text)
    
st.write('sentiment', kaif())

st.text_area('asdasd', "sdfosydf9p8u")