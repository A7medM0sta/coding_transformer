
---
### Datasets used in this project
* European Parliament Proceedings Parallel Corpus 1996-2011

| **Description**                                      | **File Size** | **Date Range**        |
|------------------------------------------------------|---------------|-----------------------|
| **Source Release** (text files)                      | 1.5 GB        | -                     |
| **Tools** (preprocessing tools and sentence aligner) | 8.6 KB        | -                     |
| **Parallel Corpus** Bulgarian-English                | 41 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Czech-English                    | 60 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Danish-English                   | 179 MB        | 04/1996 - 11/2011     |
| **Parallel Corpus** German-English                   | 189 MB        | 04/1996 - 11/2011     |
| **Parallel Corpus** Greek-English                    | 145 MB        | 04/1996 - 11/2011     |
| **Parallel Corpus** Spanish-English                  | 187 MB        | 04/1996 - 11/2011     |
| **Parallel Corpus** Estonian-English                 | 57 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Finnish-English                  | 179 MB        | 01/1997 - 11/2011     |
| **Parallel Corpus** French-English                   | 194 MB        | 04/1996 - 11/2011     |
| **Parallel Corpus** Hungarian-English                | 59 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Italian-English                  | 188 MB        | 04/1996 - 11/2011     |
| **Parallel Corpus** Lithuanian-English               | 57 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Latvian-English                  | 57 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Dutch-English                    | 190 MB        | 04/1996 - 11/2011     |
| **Parallel Corpus** Polish-English                   | 59 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Portuguese-English               | 189 MB        | 04/1996 - 11/2011     |
| **Parallel Corpus** Romanian-English                 | 37 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Slovak-English                   | 59 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Slovene-English                  | 54 MB         | 01/2007 - 11/2011     |
| **Parallel Corpus** Swedish-English                  | 171 MB        | 01/1997 - 11/2011     |
* I used many parts from it but mainly the English-French part.
* source: https://www.statmt.org/europarl/
---



# Transformer Architecture

## Introduction
The Transformer architecture revolutionized natural language processing by introducing a model that relies on attention mechanisms rather than sequential processing. It is widely used in tasks such as translation, summarization, and text generation. Below, we’ll walk through each component of the Transformer architecture step by step.

![Transformer](assets/transformer.png)

## **Explanation & Implementation**

We will explain and implement each part of the Transformer, starting from **positional encoding** and moving through the core components like **attention mechanisms**, **encoder** and **decoder blocks**, and finishing with **output processing**.

### 1. **Positional Encodings**
Transformers process sentences as sets of tokens without considering their order, so **positional encoding** is introduced to inject order information. It represents the position of each word as a vector, which is combined with word embeddings.

There are two approaches to positional encodings:

- **Fixed Positional Encoding**: This uses predefined vectors based on mathematical functions (e.g., sine and cosine). This method is used in the original Transformer paper and allows capturing relative positions effectively.
- **Learned Positional Encoding**: The model learns positional encodings during training, providing flexibility to adapt to the data. This is commonly used in state-of-the-art models.

Example formula for fixed positional encoding:
$$PE_{(pos,2i)} = sin(pos / 10000^{2i / d_{model}})$$
$$PE_{(pos,2i+1)} = cos(pos / 10000^{2i / d_{model}})$$

The embedding dimensions and the number of tokens (with a limit, e.g., 512) dictate how positional encodings are applied.

### 2. **Scaled Dot-Product Attention (SDPA)**
This is the core mechanism behind the attention in transformers. SDPA allows each word to focus on other words that are most relevant to it, enhancing contextual understanding.

In SDPA:
- **Query (Q)**: The word looking for relevant information.
- **Key (K)**: The information tags of other words.
- **Value (V)**: The actual information the words hold.

The process involves computing the similarity (dot product) between queries and keys, scaling the result, applying softmax to get attention weights, and then weighting the values accordingly. This allows the model to adjust the focus on different parts of the sentence dynamically.

Formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{{Q \cdot K^T}}{{\sqrt{d_k}}}\right) \cdot V$$

### 3. **Multi-Head Attention**
Instead of using a single attention mechanism, the Transformer applies multiple attention mechanisms (or heads) in parallel. Each head can focus on different aspects of the sentence, such as semantic or syntactic information. The outputs of these heads are concatenated and transformed to create a richer representation.

### 4. **Masked Multi-Head Attention**
Used in the decoder, masked multi-head attention ensures that the model cannot "cheat" by looking at future tokens. This is done through **causal masking**, which prevents the model from seeing tokens beyond the current position.

In addition to causal masking, **padding masks** ensure that `[PAD]` tokens, used to equalize sequence lengths, do not affect attention calculations.

### 5. **Encoder Block**
Each encoder block consists of:
- **Multi-Head Self-Attention**: Focuses on relationships within the input sequence.
- **Feed-Forward Network (FFN)**: Processes the information to produce a richer representation.
- **Residual Connections**: Shortcuts that allow information to bypass layers, aiding in training deep networks.
- **ReZero**: A variant that replaces layer normalization and skip connections with a simple weighted addition of the output.

### 6. **Decoder Block**
Similar to the encoder block, but with two key differences:
- **Masked Multi-Head Attention**: Prevents the model from looking at future tokens during training.
- **Cross Attention**: Captures relationships between the decoder input (queries) and the encoder’s output (keys and values), allowing the decoder to generate context-aware translations.

### 7. **Output Processing**
Once the decoder generates output embeddings, a **linear layer** predicts the next token by assigning a score (logit) to each word in the vocabulary. A **softmax** function converts these logits into probabilities.

* The error function, **CrossEntropyLoss**, compares the model’s predictions with the actual tokens and adjusts the model to minimize the error, allowing it to learn from its mistakes.
---















#### References:
* https://deeprevision.github.io/posts/001-transformer/