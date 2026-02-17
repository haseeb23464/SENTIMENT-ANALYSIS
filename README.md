# SENTIMENT-ANALYSIS
entiment Analysis using LSTM (TensorFlow)
📌 Project Overview

This project is a Deep Learning-based Sentiment Analysis system built using TensorFlow, NumPy, Pandas, and Scikit-learn.

The model analyzes text input and classifies it into three sentiment categories:

😊 Positive

😐 Neutral

😢 Negative

It uses Natural Language Processing (NLP) techniques such as tokenization, padding, and word embeddings along with an LSTM neural network.

🚀 Features

Text preprocessing using Tokenizer

Label encoding for multi-class classification

LSTM-based Deep Learning model

Real-time sentiment prediction

Emoji output based on prediction

🛠 Technologies Used

TensorFlow

NumPy

Pandas

scikit-learn

🧠 Model Architecture

Embedding Layer → LSTM Layer → Dense Layer → Output Layer (Softmax)

Embedding dimension: 16

LSTM units: 32

Output classes: 3

📂 Dataset

A small sample dataset is created inside the script for demonstration purposes:

Text	Sentiment
I am so happy today	Positive
I hate this weather	Negative
It is an okay day
