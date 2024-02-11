#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np 
import pickle
import pandas as pd
import streamlit as st
from PIL import Image


load = open("model.pkl","rb")
model = pickle.load(load)


def predict(competitiveness,financial_flexibility,credibility):
    prediction = model.predict([[competitiveness,financial_flexibility,credibility]])
    return prediction

def main():
    
    html_temp = """
    <div style="background-color:skyblue;padding:10px">
    <h2 style="color:black;text-align:center;">Bankruptcy Predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.markdown('A Web application for prediction of Bankruptcy')
    cc = st.selectbox('Competitiveness', ('1','0.5','0'))
    ff = st.selectbox('Financial_flexibility', ('1','0.5','0'))
    cre = st.selectbox('Credibility',('1','0.5','0'))
    if st.button('Predict'):
        result = predict(cc, ff, cre)
        if result==1:
                  st.success('The company will not go bankrupt'.format(result))
        else:
            st.success('The company will go bankrupt'.format(result))
        
if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:




