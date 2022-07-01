from nltk.tokenize import word_tokenize
import pandas as pd
import gensim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk

#Load in data
df = pd.read_json("MovieReactionDS_medium.json")
df['output'] = df['output'].map({'negative': 0, 'positive': 1})

# Tokenize input data
df['tokenized_sents'] = df['input'].apply(word_tokenize)

# Train- test split
X_train, X_test, y_train, y_test = train_test_split(
    df['tokenized_sents'], df['output'], test_size=0.4)

# Create Word2Vector model from X_Train
w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=100,
                                   window=5,
                                   min_count=2,
                                   epochs=1000)

# Add Vectors from model to X_train_vect and X_test_vect
words = set(w2v_model.wv.index_to_key)
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                        for ls in X_test])


# Create averages for each word vector and save them in X_train_vect_avg and X_test_vect_avg
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))

X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))

# init Logistic regression and fit to vectors and lables
lr_clf = LogisticRegression()
lr_clf.fit(X_train_vect_avg, y_train.values)

# print accuracy and classification report
print('Test Accuracy: ', lr_clf.score(X_test_vect_avg, y_test.values))

print(classification_report(y_test,
                            lr_clf.predict(X_test_vect_avg)))
