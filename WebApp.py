# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:19:42 2023

@author: acer
"""

import streamlit as st
import pickle as p
from skimage.transform import resize
from skimage.io import imread
Categories= ['droppy', 'normal_1']

model = p.load(open('img_model.p','rb'))

def predict (url):
        img=imread(url)
        img_resize=resize(img,(150,150,3))
        l=[img_resize.flatten()]
        probability=model.predict_proba(l)  
        return probability, Categories[model.predict(l)[0]]
st.header('Droopy Face Detection Web')
URL= st.text_input('Masukkan URL')
#url= https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.jems.com%2Fpatient-care%2Fdifferentiating-facial-weakness-caused-b%2F&psig=AOvVaw0UYM6_sOpZOK5D0UDvDA3_&ust=1673328201901000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCPCC_brfufwCFQAAAAAdAAAAABAE''
with st.expander("See explanation"):
    st.write("The chart above shows some numbers I picked for you. I rolled actual dice for these, so they're *guaranteed* tobe random."
    )
    st.image("https://static.streamlit.io/examples/dice.jpg")
try:  
  prob, category = predict(URL)
  for ind,val in enumerate(Categories):
      st.write (f'{val} = {prob[0][ind]*100}%')
except:
    pass
