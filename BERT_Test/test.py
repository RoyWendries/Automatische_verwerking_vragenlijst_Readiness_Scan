from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import gensim
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_json("MovieReactionDS_medium.json")

# CONTROLEREN OP NULL WAARDES
'''df.isna().sum()

# LEESTEKENS
df["Gecleand"] = df['input'].str.replace('[^\w\s]', '')

# De kolom `Gecleand` alles in kleine letters zetten
df['Gecleand'] = df['Gecleand'].str.lower()

# stopwoorden eruit halen
stop = stopwords.words('english')
df['Gecleand'] = df['Gecleand'].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in (stop)]))

# Positive eruit halen

#data = df[df['output']== 'positive']
data = df[['Gecleand', 'output']]
'''

df['output'] = df['output'].map({'negative': 0, 'positive': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['input'], df['output'], test_size=0.2)

w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size=100,
                                   window=5,
                                   min_count=2)

words = set(w2v_model.wv.index_to_key)
X_train_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                         for ls in X_train])
X_test_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in words])
                        for ls in X_test])


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

lr_clf = LogisticRegression()
lr_clf.fit(X_train_vect_avg, y_train.values)

print('Test Accuracy: ', lr_clf.score(X_test_vect_avg, y_test.values))

print(classification_report(y_test,
                            lr_clf.predict(X_test_vect_avg)))


'''# Tokenize
corpus = []
for i in sent_tokenize(data):
    temp = []
    for j in word_tokenize(i):
        temp.append(j)
    corpus.append(temp)

print(corpus)
# Creating Word2Vec
model = Word2Vec(
    sentences=corpus, window=10)
'''
