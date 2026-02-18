import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing import sequence
import streamlit as st


## load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## load the model
model = load_model('simple_rnn.h5')


# step-2 Helper Function
# Function to decode the reviews
def decode_review(encoded_review):
    return' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])


# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        index = word_index.get(word)

        if index is not None and index < 10000:
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)  # unknown token

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review



### Step 3: creating our prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'
    return sentiment,prediction[0][0]


## Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter A Movie Review to classify it as positive or negative.')

#user input
user_input = st.text_area('Movie Review')

if st.button('classify'):
    preprocessed_input = preprocess_text(user_input)

    #make prediction
    prediction = model.predict(preprocessed_input)
    Sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'

    # Display the result 
    st.write(f'Sentiment:{Sentiment}') 
    st.write(f'Prediction score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review')

