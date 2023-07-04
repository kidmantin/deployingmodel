import streamlit as st
import training

# def kaif():
#     st.write("kaif received")
#     return True

st.title('AYO CHILL')
file = st.file_uploader('audio of command')
button1 = st.button('chill?')

if button1:
    st.write(file)

st.audio(file)

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