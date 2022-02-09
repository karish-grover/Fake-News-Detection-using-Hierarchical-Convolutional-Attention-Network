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
- **BERT** (Bidirectional Encoder Representations from Transformers), **RoBERTa** (Robustly Optimized BERT Pretraining Approach). Huggingface implementation of the **bert-base-cased** and **roberta-base** model finetuned using the **AdamW** optimizer, with a batch size of 8 for 3 epochs on NVIDIA Tesla V100 GPU.

More details about these baselines are mentioned in the [paper]().
<center>
<table align="center">
<tr><td>

| **Model**             |  **Prec**   |  **Rec**    |   **F1**   | 
| --------------------- |------------ |------------ |------------| 
| LR *w/* word-tfidf    |   0.9694    |   0.9643    |   0.9669   | 
| LR *w/* char-tfidf    |   0.9613    |   0.9650    |   0.9631   | 
| LR *w/* ngram-tfidf   |   0.9718    |   0.9296    |   0.9502   | 
| RF *w/* word-tfidf    |   0.9121    |   0.9675    |   0.9390   | 
| RF *w/* char-tfidf    |   0.9675    |   0.9684    |   0.9680   | 
| DT *w/* word-tfidf    |   0.9517    |   0.9642    |   0.9579   | 
| DT *w/* char-tfidf    |   0.9642    |   0.9777    |   0.9709   | 
| LSTM                  |   0.9322    |   0.9384    |   0.9353   | 
| GRU                   |   0.9637    |   0.9202    |   0.9414   | 
| GRU *w/o* Dropout     |   0.9274    |   0.9422    |   0.9348   | 
| GRU *w/* trainEmb     |   0.9522    |   0.9609    |   0.9566   | 
| BiGRU                 |   0.9326    |   0.9362    |   0.9344   | 
| BiGRU *w/o* Dropout   |   0.9656    |   0.9149    |   0.9396   | 
| BiGRU *w/* trainEmb   |   0.9646    |   0.9439    |   0.9542   | 

</td><td>

| **Model**             |  **Prec**   |  **Rec**    |   **F1**   |
| --------------------- |------------ |------------ |------------|
| CNN                   |   0.9389    |   0.9221    |   0.9304   |
| CNN *w/* trainEmb     |   0.9742    |   0.9222    |   0.9475   |
| CNN *w/o* Dropout     |   0.9680    |   0.9358    |   0.9516   |
| RCNN                  |   0.9689    |   0.9018    |   0.9341   |
| RCNN *w/* trainEmb    |   0.9145    |   0.9642    |   0.9387   |
| RCNN *w/o* Dropout    |   0.9274    |   0.9400    |   0.9336   |
| RCNN *w/* UniGRU      |   0.9584    |   0.9317    |   0.9449   |
| RCNN *w/* BiLSTM      |   0.9665    |   0.8964    |   0.9301   |
| RoBERTa               |   0.9781    |   0.9753    |   0.9789   |
| BERT                  |   0.9781    |   0.9772    |   0.9802   |
| HCAN *w/o* Sent.E.    |   0.9410    |   0.9335    |   0.9373   |
| HCAN *w/o* CNN        |   0.9877    |   0.9605    |   0.9739   |
| HCAN *w/* UniGRU      |   0.9632    |   0.9869    |   0.9749   |
| **HCAN (Ours)**       | **0.9891**  | **0.9835**  | **0.9863** |

</td></tr> </table>
</center>
