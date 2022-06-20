import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')


df = pd.read_json('MovieReactionDS_medium.json')
df['output'] = df['output'].map({'negative': 0, 'positive': 1})

batch_1 = df[:2000]
# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (
    ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = batch_1['input'].apply(
    (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512)))

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
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels)
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)
print(lr_clf.score(test_features, test_labels))
print(classification_report(test_labels,
                            lr_clf.predict(test_features)))
