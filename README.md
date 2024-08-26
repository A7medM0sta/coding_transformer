# Transformer Architecture
## Introduction
![Transformer](assets/transformer.png)
## **Explanation & Implementation**
In this section, we will explain and implement our transformer step by step. We have already learned about the **embedding layer**, which maps `token ids` to `vectors`.

This layer is shown as the **input and output embeddings** in the transformer picture above 👆👆.

So, we will start our explanation from **positional encoding**.

### **Positional Encodings**
The position of each word in a sentence affects its meaning. For example, "I love pizza" 🍕 and "Pizza love I" have the same words, but different orders and meanings.

We can use positional encoding to represent the position of each word as a vector, similar to `word embeddings`. This way, we can teach a computer to understand the order and the context of the words in a sentence.

There are two ways of creating positional encoding vectors:

- Fixed positional encoding: The vectors are predefined and fixed. Formulas such as sine and cosine functions is used:

$$PE_{(pos,2i)} = sin(pos / 10000^{2i / d_{model}})$$
$$PE_{(pos,2i+1)} = cos(pos / 10000^{2i / d_{model}})$$

where $pos$ is the position, $i$ is the dimension and $d_{model}$ is the size of embedding dimension (number of elements in positional encoding vector). This way captures relative distances and handle variable length sentences. *`[This is what the original transformer paper did]`*.

- Learned positional encoding: The vectors are randomly initialized and learned by the model. An embedding layer is used with different embeddings for each position. This way the model learns the positional information to add to the data enabling it to potentially capture more complex patterns. *`[This is what most state-of-the-art models do]`*.

Note: When we are using learned positional encodings we are going to need to put a limit on the number of tokens our model can take in. This is because the position encoding layer is going to have an embedding matrix with as many rows as our max number of tokens. Its number of columns should also match those of the word embeddings because we are going to add `+` them together, this way we get a new vector that contains both the meaning and the position of each word.

In this tutorial, we use the positional encoding layer from the `pre_trained_model`. It has a token limit of 512, and our model `TOKEN_LIMIT=350` so we are good to go.
