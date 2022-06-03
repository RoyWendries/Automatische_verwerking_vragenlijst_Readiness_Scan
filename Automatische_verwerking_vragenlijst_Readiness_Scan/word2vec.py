from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

tekst = open("C:\\Users\\shirl\\Desktop\\alice_in_wonderland.txt")
test = tekst.read()
test = str(test)
f = test.replace("\n", " ")

data = []

for i in sent_tokenize(f):
    temp=[]
    for j in word_tokenize(i):
        temp.append(j.lower())
    
    data.append(temp)
    
#Skip gram model
model1 = gensim.models.Word2Vec(data,min_count = 1, vector_size = 100, window=5, sg=1)
skip = model1.wv.similarity("alice", "wonderland")
skip = str(skip)
print("Simulariteit skip gram: 'Alice' en 'wonderland': " + skip)
test2 = model1.wv.similarity("alice", "machines")
test2 = str(test2)
print("Simulariteit skip gram 'alice' en 'machines'" + test2)

#CBOW model
model2 = gensim.models.Word2Vec(data, min_count = 1,
                              vector_size = 100, window = 5)

cbow = model2.wv.similarity("alice", "wonderland")
cbow = str(cbow)
print("Simulariteit CBOW: 'Alice' en 'wonderland': " + cbow)

