import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

# load dataset
df = pd.read_json('MovieReactionDS_small.json')
df['output'] = df['output'].map({'negative': 0, 'positive': 1})

# uncomment for larger datasets
'''# Create 20 batches with 500 samples each
batches = []
n = 0
while n < 20:
    batch = df.sample(500)
    batches.append(batch)
    n += 1
'''
# Selecting distilbert
model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# change batch_1 to 'df.sample(500)' for larger datasets
batch_1 = df
print('\nBatch: 1')

# tokenize input data
tokenized = (batch_1['input'].apply(
    (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512))))
max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

# Pad all inputdata and create attention mask
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)

# Create tensors for input ids and attention mask
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

# Make model using pretrained BERT with input ids and attention mask as inputs
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

# extract features from the BERT trained model
features = last_hidden_states[0][:, 0, :].numpy()
labels = batch_1['output']

# uncomment for larger datasets
'''#iteraties through batches and adds result to Features and Labels to save on RAM Space
num = 2
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

    features = np.concatenate([features, feature])
    labels = np.concatenate([labels, label])
'''
# Train test split
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.4)

# init Logistic regression and fit to train features and lables
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

# print accuracy and classification report
print('Test Accuracy: ', lr_clf.score(test_features, test_labels))
print(classification_report(test_labels,
                            lr_clf.predict(test_features)))
