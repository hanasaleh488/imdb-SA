import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Load the pre-trained model and tokenizer
model_path = os.path.join('models', 'model.keras')
model = tf.keras.models.load_model(model_path)

tokenizer_path = os.path.join('models', 'tokenizer.pickle')
tokenizer = pickle.load(open(tokenizer_path, 'rb'))

# Preprocess user input text
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=model.input_shape[1], padding='post', truncating='post')
    return padded

# Title
st.title("IMDB Sentiment Classifier")

# Input field
user_input = st.text_area("Write your review here:", height=200)

# Prediction
if st.button('Predict'):
    if user_input.strip():
        processed_input = preprocess_text(user_input)
        prediction = model.predict(processed_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

        # Display result
        if sentiment == "Positive":
            st.success(f"The review is **Positive**.")
        else:
            st.error(f"The review is **Negative**.")
    else:
        st.warning("Please write a review before clicking Predict!")