import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')


df = pd.read_json('MovieReactionDS.json')
df['output'] = df['output'].map({'negative': 0, 'positive': 1})


batches = []
n = 0
while n < 20:
    batch = df.sample(300)
    batches.append(batch)
    n += 1

# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

batch_1 = df.sample(300)

tokenized = (batch_1['input'].apply(
    (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512))))
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:, 0, :].numpy()
labels = batch_1['output']


num = 1
accuracy = []
for batch in batches:

    print('\nBatch: ', num)
    num += 1
    tokenized = (batch['input'].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512))))
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)

    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    feature = last_hidden_states[0][:, 0, :].numpy()
    label = batch['output']

    print(feature, label)

    features = np.concatenate([features, feature])
    labels = np.concatenate([labels, label])
    print(features, labels)

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.4)

lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
print('Test Accuracy: ', lr_clf.score(test_features, test_labels))

'''
    lr_clf = SGDClassifier(loss="log")
    lr_clf.partial_fit(train_features, train_labels,
                       classes=np.unique(train_labels))
    print('Test Accuracy: ', lr_clf.score(test_features, test_labels))
    accuracy.append(lr_clf.score(test_features, test_labels))


def Gem_Acc(list):
    return sum(list) / len(list)


print('Gemiddelde accuracy: ', Gem_Acc(accuracy))
'''
print(classification_report(test_labels,
                            lr_clf.predict(test_features)))
