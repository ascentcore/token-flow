import bcolz
import pickle
import numpy as np

words = []
idx = 0
word2idx = {}
vectors = []

with open('src/embeddings/glove.6B.50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(float)
        vectors.append(vect)
    
pickle.dump(words, open('src/embeddings/glove.6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open('src/embeddings/glove.6B.50_idx.pkl', 'wb'))
pickle.dump(vectors, open('src/embeddings/glove.6B.50_vectors.pkl', 'wb'))

def get_embeddings():
    words = pickle.load(open('src/embeddings/glove.6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open('src/embeddings/glove.6B.50_idx.pkl', 'rb'))
    vectors = pickle.load(open('src/embeddings/glove.6B.50_vectors.pkl', 'rb'))

    embeddings = {w: vectors[word2idx[w]] for w in words}

    return embeddings
