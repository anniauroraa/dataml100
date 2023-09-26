import random
import numpy as np

vocabulary_file='C:\\code\\repositories\\dataml100\\word_embeddings.txt'

def get_nearest_neighbors(vector, n):
    
    # Calculate Euclidean distances
    distances = np.linalg.norm(W - vector, axis=1)

    # Get the indices of the 3 closest vectors
    k = n
    nearest_indices = np.argpartition(distances, k)[:k]
    sorted = np.argsort(distances[nearest_indices])
    sorted_nearest_indices = nearest_indices[sorted]

    return sorted_nearest_indices, distances

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}


# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)
    
# Main loop for analogy
while True:
    input_term = input("\nEnter three words (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        words = input_term.rstrip().split('-')

        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")

        # Calculate location for the R in analogy "X is to Y as Z is to R”
        x = W[vocab[words[0]]]
        y = W[vocab[words[1]]]
        z = W[vocab[words[2]]]

        search_point = z + (y - x)

        nearest_indices, distances = get_nearest_neighbors(search_point, 2)

        for x in nearest_indices:
            word = ivocab[x]
            distance = distances[x]
            print("%35s\t\t%f" % (word, distance))