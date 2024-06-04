# Attention Models:
Attention mechanisms allow models to focus on different parts of the input sequence for each step of the output sequence, which is crucial for tasks like machine translation. The document introduces and explains two main types of attention:

1. Scaled Dot-Product Attention: This type of attention calculates scaled dot products between the query and keys, followed by a softmax function to obtain the attention weights, which are then used to weight the corresponding values​​.

2. Multi-Head Attention: This extension of the attention mechanism allows the model to focus on different parts of the sequence simultaneously using multiple attention "heads." Each head processes a linearly projected version of the queries, keys, and values, and their results are concatenated and projected again to form the output​​.

# Transformers:
Transformers are a novel architecture that relies solely on attention mechanisms, completely eliminating the use of recurrent neural networks (RNNs) and convolutional neural networks (CNNs). The key points about transformers are:

1. Architecture: Transformers follow an encoder-decoder structure, where both the encoder and decoder consist of a stack of identical layers combining self-attention mechanisms and fully connected feed-forward networks. Self-attention allows each position in the input sequence to connect with all other positions in each layer of the encoder and decoder​​.

2. Self-Attention: This is a form of attention that relates different positions of a single sequence to compute a representation of the sequence. It is particularly effective at modeling long-term dependencies and improving parallelization during training, which is a significant advantage over traditional RNNs and CNNs​​.

3. Advantages: Transformers allow greater parallelization and thus reduce training time. They also handle long-term dependencies better than RNNs and CNNs and have been shown to achieve superior results in tasks like machine translation and other natural language processing tasks​​.