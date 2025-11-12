# Musical-Instruments-Reviews-Sentiment-Analysis
## Dataset
Source: Kaggle - Amazon Musical Instruments Reviews by Eswar Chand
https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews/data?select=Musical_instruments_reviews.csv

NLP project for reviews sentiment analysis. Implemented TF-IDF, Word2Vec and GloVe with logistic regression and DistilBERT for classification. Evaluated models using Precision, Recall and F1-score.

## Data preprocessing
First, load and preprocess the data. Reviews with rating ≥4 are positive (label 1) and label 0 for the rest. 

## TF-IDF
Vectorize the text using a TF–IDF representation  and train a simple logistic regression classifier.

## Word2Vec
Next, train a Word2Vec model to get word embeddings, and represent each review by the average of its word vectors. These averaged vectors are then fed to another logistic regression.

## GloVe
I also tried GloVe pre-trained word embeddings. Load GloVe (Twitter data, 100-dimensional) via Gensim’s API and average the word vectors similarly. Then train another logistic regression.

## DistilBERT
Finally, use DistilBERT, a transformer-based language model, which is lighter than BERT that has become state-of-the-art in NLP. Load a pre-trained DistilBERT and add a small classification layer on top.

- Tokenize the text data: convert each review into input IDs and attention masks.

- Create Dataset objects for easy batching. Each item will contain input_ids, attention_mask, and labels.

- Define a simple classifier: DistilBERT encoder + dropout + linear layer. Following standard practice, I took the embedding of the first token as the sentence representation. A dropout layer is applied for regularization, then a linear layer to two logits.

- Train the model with a few epochs using cross-entropy loss and the AdamW optimizer and then evaluate it.

## Overall
- Winner (overall F1): DistilBERT (0.945) — best balance of precision (0.912) and recall (0.981). It catches most positives while making fewer false positives than the others.

- TF-IDF / GloVe (F1 = 0.938 / 0.937) — extremely high recall (~0.998) but lower precision (~0.883–0.885). They flag almost everything positive but at the cost of more false positives.

- Word2Vec (F1 = 0.935) — slightly lower recall than TF-IDF/GloVe and similar precision.

If care about catching every positive (don’t want to miss happy customers): TF-IDF/GloVe are great because of their near-perfect recall, accepting extra false positives.

If care about being right saying “positive” (fewer false positives): DistilBERT (or BERT) is better (highest precision).