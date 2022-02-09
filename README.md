# Fake News Detection using Hierarchical Convolutional Attention Network

This project was carried out as a part of the Natural Language Processing - NLP (CSE556) course, at **IIIT Delhi**. 

## Abstract
In today’s digital era, social media platforms’ popularity has increased exponentially, leading to an increase in the amount of unreliable information online. Verifying the integrity of a particular piece of news is not easy for any end-user. In this paper, our end goal is to design an efficient model to examine the reliability of a news piece. This paper addresses the problem of Fake News classification, given only the news piece content and the author name. We present, **Hierarchical Convolutional-Attention Network (HCAN)** composed of attention-enhanced word-level and sentence-level encoders and a CNN to capture the sequential correlation. Extensive experiments show that HCAN outperforms the state-of-the-art baselines models on a Kaggle dataset. 


## Dataset and Preprocessing
We use the Kaggle [Fake news detection dataset](https://www.kaggle.com/c/fake-news/data) for our task. There are three attributes in the dataset, `author`, `title`, and `text`. We concatenate all the three features in order to make the final predictions. This is because the credibility of an author plays a very crucial role in determining the reliability of a news piece. Further, many times a news title has a particular writing style or phrases, and by detecting such patterns one can be more certain about a news article. Next, we remove the stop words and punctuations to further process our dataset. All the experiments are performed on a `80:20` train-test split.


## Baselines
We evaluate the dataset collected on several baselines, as listed in the table below. We evaluate the performance of these baselines using F1 score, Recall, and Precision. The implementation of these baselines has been released. We implement three types of baseline models, Simple Linear Classification Models, Deep Neural Network Models, and Pretrained Language Models. All the baselines have been implemented in the notebooks:- [BERT-RoBERTa.ipynb](BERT-RoBERTa.ipynb), [LR-MNB-DT.ipynb](LR-MNB-DT.ipynb), and [NN_Baselines.ipynb](NN_Baselines.ipynb).


- **LR** (Logistic Regression), **DT** (Decision Trees), and **RF** (Random Forest). Trained using TFIDF vectors of the input text. 
- **CNN** (Convolutional Neural Networks). 1D Convolutional layer with kernel size 3, followed by max pooling and a fully connected Dense layer. All deep neural models have been trained for a maximum input length of 70.
- **RNN** (Recurrent Neural Networks), **LSTM** (Long Short Term Memory cells), **GRU** (Gated Recurrent Networks), **Bi-RNN** (Bi-directional Re-current Neural Networks). Respective RNN followed by dropout and fully connected layers.
- **RCNN** (Recurrent Convolutional Neural Networks). Uses Bidirectional GRU to encode the Glove embeddings of the tokens, 1D Convolutional layer, followed by a max pooling and dropout layer. 
- **BERT** (Bidirectional Encoder Representations from Transformers), **RoBERTa** (Robustly Optimized BERT Pretraining Approach). Huggingface implementation of the `bert-base-cased` and `roberta-base` model finetuned using the `AdamW` optimizer, with a batch size of 8 for 3 epochs on `NVIDIA Tesla V100 GPU`.

More details about these baselines are mentioned in the [paper](NLP_Project.pdf).
<center>
 <table align="center">
<tr><td>

| **Model**             |  **Prec**   |  **Rec**    |   **F1**   | 
| --------------------- |------------ |------------ |------------| 
| LR *w/* `word-tfidf `   |   0.9694    |   0.9643    |   0.9669   | 
| LR *w/* `char-tfidf `   |   0.9613    |   0.9650    |   0.9631   | 
| LR *w/* `ngram-tfidf`   |   0.9718    |   0.9296    |   0.9502   | 
| RF *w/* `word-tfidf `   |   0.9121    |   0.9675    |   0.9390   | 
| RF *w/* `char-tfidf `   |   0.9675    |   0.9684    |   0.9680   | 
| DT *w/* `word-tfidf `   |   0.9517    |   0.9642    |   0.9579   | 
| DT *w/* `char-tfidf `   |   0.9642    |   0.9777    |   0.9709   | 
| LSTM                  |   0.9322    |   0.9384    |   0.9353   | 
| GRU                   |   0.9637    |   0.9202    |   0.9414   | 
| GRU *w/o* `Dropout`    |   0.9274    |   0.9422    |   0.9348   | 
| GRU *w/* `trainEmb`     |   0.9522    |   0.9609    |   0.9566   | 
| BiGRU                 |   0.9326    |   0.9362    |   0.9344   | 
| BiGRU *w/o* `Dropout`   |   0.9656    |   0.9149    |   0.9396   | 
| BiGRU *w/* `trainEmb`   |   0.9646    |   0.9439    |   0.9542   | 

</td><td>

| **Model**             |  **Prec**   |  **Rec**    |   **F1**   |
| --------------------- |------------ |------------ |------------|
| CNN                   |   0.9389    |   0.9221    |   0.9304   |
| CNN *w/* `trainEmb`     |   0.9742    |   0.9222    |   0.9475   |
| CNN *w/o* `Dropout`     |   0.9680    |   0.9358    |   0.9516   |
| RCNN                  |   0.9689    |   0.9018    |   0.9341   |
| RCNN *w/* `trainEmb`    |   0.9145    |   0.9642    |   0.9387   |
| RCNN *w/o* `Dropout`    |   0.9274    |   0.9400    |   0.9336   |
| RCNN *w/* `UniGRU`      |   0.9584    |   0.9317    |   0.9449   |
| RCNN *w/* `BiLSTM`      |   0.9665    |   0.8964    |   0.9301   |
| RoBERTa               |   0.9781    |   0.9753    |   0.9789   |
| BERT                  |   0.9781    |   0.9772    |   0.9802   |
| HCAN *w/o* `Sent.E.`    |   0.9410    |   0.9335    |   0.9373   |
| HCAN *w/o* `CNN`        |   0.9877    |   0.9605    |   0.9739   |
| HCAN *w/* `UniGRU`      |   0.9632    |   0.9869    |   0.9749   |
| **HCAN (Ours)**       | **0.9891**  | **0.9835**  | **0.9863** |
</td></tr> </table>
</center>
 
 
## Model Architecture

The model architecture developed has been shown in the following figure:

<p align="center">
  <img align = "center" width="491" alt="Screenshot 2022-02-09 at 5 12 15 PM" src="https://user-images.githubusercontent.com/64140048/153193237-e06228d3-6507-4217-86d5-506a50078f7d.png">
</p>

The notebook [HCAN.ipynb](HCAN.ipynb) contains the model implementation. 



## Results and Analysis

It can be seen from the table that our system HCAN outperforms all the
baseline models by a decent margin and gets an F1 score of **0.9856**.
We further analyse the effect of the Convolutional layer and the
hierarchical sentence encoder on the model performance. It can be seen
that on removing the hierarchy i.e. by considering only the word-level
encoder i.e. an attention enhanced bi-directional network, the accuracy
degrades the most. Further, removing the CNN layer from the model also
leads to poor performance. 


<center>
<table align="center">
<tr><td>
<img width="410" alt="graph" src="https://user-images.githubusercontent.com/64140048/153190777-b494d7cc-90ef-491c-9fc3-9b6156a67046.png"> 
</td><td>
<img width="410" alt="max_len_graph" src="https://user-images.githubusercontent.com/64140048/153190790-7ab06e0f-6097-42f2-87fe-6df142f1e25f.png">
</td></tr> </table>
</center>

We also analyse the effect of the CNN kernel size and word-encoder max-lengths on the testing and training accuracies.
  
The hierarchical structure of our model, enhanced by a CNN, outperforms
even the state-of-the-art pretrained language models like BERT and
RoBERTa. This can be related back to our starting motivation to model
more important parts of a news piece to make the prediction. Our
motivations are supported by the experimental results.



