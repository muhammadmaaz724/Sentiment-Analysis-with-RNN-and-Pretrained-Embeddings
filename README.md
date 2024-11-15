# Sentiment Analysis with RNN and Pretrained Embeddings

This project demonstrates a sentiment analysis system using a Recurrent Neural Network (RNN) with GloVe pretrained word embeddings. The model classifies reviews as either **positive** or **negative** based on their text content.

---

## Overview
This project utilizes a bidirectional LSTM-based RNN for sentiment classification. By leveraging **GloVe** pretrained embeddings, the model can classify text reviews as either positive or negative. The focus is on applying deep learning techniques for Natural Language Processing (NLP) to analyze textual data.

---

## Dataset
The dataset consists of text reviews labeled as either **positive** or **negative**. Each review is preprocessed, tokenized, and converted into padded sequences. The dataset enables the training of the model for binary sentiment classification.

---

## Model Architecture
The sentiment analysis model follows this architecture:
- **Embedding Layer**: Uses GloVe pretrained embeddings (300-dimensional).
- **Bidirectional LSTM**: Captures context in both forward and backward directions.
- **Fully Connected Layers**: Used for binary classification of sentiment.
- **Dropout**: Prevents overfitting by randomly disabling neurons during training.

The model aims to predict whether a review is **positive** or **negative** based on its content.

---

## Pretrained Embeddings
The model utilizes **GloVe (Global Vectors for Word Representation)** embeddings, which are pretrained word vectors designed to capture the meaning of words based on their context. These embeddings are 300-dimensional and can be used to enhance model accuracy for text classification tasks.

GloVe embeddings can be downloaded from [the official GloVe website](https://nlp.stanford.edu/projects/glove/).

---

## Setup Instructions

### Prerequisites
1. Python 3.7+
2. Required libraries:
   - TensorFlow
   - NumPy
   - Pandas
   - Pickle


