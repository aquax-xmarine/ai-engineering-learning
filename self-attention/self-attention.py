import numpy as np

# Function to build simple embeddings
def build_embeddings(word):
    return np.random.rand(4)

# Example sentence
sentence = "In this tutorial I will show you how to build embeddings and the self attention mechanism."

# Splitting the sentence into words
words = sentence.split()

# Generating embeddings for each word
embeddings = [build_embeddings(word) for word in words]


# Simplified softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Simplified self-attention mechanism
def self_attention(embeddings):
   
    # Building Q, K, V
    Q = np.array(embeddings)
    K = np.array(embeddings)
    V = np.array(embeddings)
    
    # Calculating the attention scores
    scores = np.dot(Q, K.T)
    
    # Applying softmax function to get the attention weights
    attention_weights = softmax(scores)
    
    # Measuring the weighted values
    weighted_values = np.dot(attention_weights, V)
    return attention_weights, weighted_values

# Running self-attention
attention_weights, weighted_values = self_attention(embeddings)

# Displaying the attention weights
print("Original sentence: ", sentence)
for i, word in enumerate(words):
    print("For the word", word)
    top_3 = sorted(range(len(words)),
                   key=lambda j: attention_weights[i][j],
                   reverse=True)[:3]
    print(top_3)
    for j in top_3:
        print(f"Attention for: {words[j]}: {attention_weights[i][j]}")

# Explanation of attention relationships
print("Explanation")
for i, word in enumerate(words):
    sorted_indices = np.argsort(-attention_weights[i])
    second_max_attention = sorted_indices[1]  # Ignoring the first (self-attention)
    print(
        f"The word '{word}' is most attended by '{words[second_max_attention]}'"
    )
