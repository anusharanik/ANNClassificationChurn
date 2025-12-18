# IMDB Movie Review Sentiment Analysis (Simple RNN)

This project demonstrates the implementation of a **Simple Recurrent Neural Network (RNN)** for **binary sentiment classification** on IMDB movie reviews. The focus is on understanding **sequence modeling, text preprocessing, and end-to-end deployment** using Streamlit.

---

## 🎯 Problem Statement

Given a movie review written in natural language, predict whether the sentiment expressed is **Positive** or **Negative**.

---

## 🧠 Why Simple RNN?

- Designed to handle **sequential data**
- Captures **word order and contextual information**
- Chosen intentionally to understand **core RNN concepts** before moving to advanced architectures like LSTM or GRU

---

## ⚙️ Implementation Highlights

### 1. Text Preprocessing
- Uses the **IMDB word index** provided by Keras
- Converts raw text into integer sequences
- Pads sequences to a fixed length of **500 tokens** to maintain uniform input shape

```python
sequence.pad_sequences(encoded_review, maxlen=500)

2. Model Architecture

Embedding layer for word vector representation
SimpleRNN layer to process sequences
Dense output layer with sigmoid activation for binary classification
Model is pre-trained and stored as: simpleRNNimdb.h5

3. Prediction Pipeline

User input → tokenization → padding
Model outputs probability score
Threshold of 0.5 used for sentiment classification
sentiment = 'Positive' if prediction > 0.5 else 'Negative'

🌐 Deployment Strategy

Built as a Streamlit web application
Uses relative file paths to support cross-platform execution
Fully compatible with Streamlit Cloud (Linux)

▶️ How to Run
pip install -r requirements.txt
streamlit run main.py

Key Learnings
Differences between ANNs and RNNs in handling sequential data
Importance of padding and tokenization in NLP pipelines
Challenges of Simple RNNs such as vanishing gradients
Practical experience deploying ML models using Streamlit

Future Improvements
Replace Simple RNN with LSTM or GRU
Improve preprocessing using subword tokenization
Add model confidence visualization
Train model with custom dataset for improved accuracy