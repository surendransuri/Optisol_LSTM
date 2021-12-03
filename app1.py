#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[9]:


app1= Flask(__name__)


# In[10]:


trans=pickle.load(open("Tokenize.pkl","rb"))


# In[11]:


@app1.route('/')
def home():
    return render_template('CovidTweet.html')


# In[12]:


def ValuePredictor(to_predict):
    loaded_model = pickle.load(open("LSTMModel.pkl", "rb")) 
    re= np.argmax(loaded_model.predict(to_predict), axis=1)
    return re 


# In[13]:


@app1.route('/Result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form["Tweet Message"]
        pred=[to_predict_list]
        tr=trans(pred)
        padded=pad_sequences(tr, maxlen=47, padding='post')
        result = ValuePredictor(padded)         
        if int(result)== 1: 
            prediction = "Neutral"
        elif int(result)== 2: 
            prediction = "Positive"
        else:
            prediction = "Negative"
        return render_template("Result.html", prediction = prediction)


# In[14]:


if __name__ == "__main__":
    app1.run(debug=True)


# In[ ]:




