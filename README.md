# Fake News Detection using Hierarchical Convolutional Attention Network

This project was carried out as a part of the Natural Language Processing - NLP (CSE556) course project, at IIIT Delhi. 

# Abstract
In today’s digital era, social media platforms’ popularity has increased exponentially, lead- ing to an increase in the amount of unreliable information online. Verifying the integrity of a particular piece of news is not easy for any end-user. In this paper, our end goal is to design an efficient model to examine the reliability of a news piece. This paper addresses the problem of Fake News classification, given only the news piece content and the author name. We present, **Hierarchical Convolutional-Attention Network (HCAN)** composed of attention-enhanced word-level and sentence-level encoders and a CNN to capture the sequential correlation. Extensive experiments show that HCAN outperforms the state-of-the-art baselines models on a Kaggle dataset. 


# Baselines
We evaluate the dataset collected on several baselines, as listed in the table below. We evaluate the performance of these baselines using F1 score, Recall, and Precision. The implementation of these baselines has been released. We implement three types of baseline models, Simple Linear Classification Models, Deep Neural Network Models, and Pretrained Language Models.


- **LR** (Logistic Regression), **DT** (Decision Trees), and **RF** (Random Forest). Trained using TFIDF vectors of the input text. 
- **CNN** (Convolutional Neural Networks). 1D Convolutional layer with kernel size 3, followed by max pooling and a fully connected Dense layer. All deep neural models have been trained for a maximum input length of 70.
- **RNN** (Recurrent Neural Networks), **LSTM** (Long Short Term Memory cells), **GRU** (Gated Recurrent Networks), **Bi-RNN** (Bi-directional Re-current Neural Networks). Respective RNN followed by dropout and fully connected layers.
- **RCNN** (Recurrent Convolutional Neural Networks). Uses Bidirectional GRU to encode the Glove embeddings of the tokens, 1D Convolutional layer, followed by a max pooling and dropout layer. 
- **BERT** (Bidirectional Encoder Representations from Transformers), **RoBERTa** (Robustly Optimized BERT Pretraining Approach). Huggingface implementation of the \texttt{bert-base-cased} and \texttt{roberta-base} model finetuned using the \texttt{AdamW} optimizer, with a learning rate of \texttt{4e-5} and batch size of 8 for 3 epochs on NVIDIA Tesla V100 GPU.
