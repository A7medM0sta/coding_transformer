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

**Before we understand Multi-Head & Masked Multi Head Attention we need to understand Scaled Dot-Product Attention**

#### Scaled Dot-Product Attention (SDPA)

Imagine a vibrant party teeming with words, each eager to understand its peers. This is the essence of Scaled Dot-Product Attention (SDPA), the heart of the Transformer, a powerful language model. But before we hit the dance floor, let's meet the key players:

**Word Embeddings:** Think of these as name tags at the party, each word getting a unique vector representing its meaning. But in sentences like "Do what is Right" and "Shift to your Right," the word "Right" has the same tag despite differing contexts.

**Enter SDPA, the party game changer!** It introduces three roles:

- **Query (Q):** The curious word asking, "Who has the information I need?"
- **Key (K):** Like a name tag, revealing relevant skills or knowledge.
- **Value (V):** The actual hidden talent or information the word possesses.

Now, picture each word comparing its query (e.g., "Who has the information I need?") with everyone's keys (e.g The information they have). The more relevant the key's information, the higher the "attention score" it gets. Think of it as noticing someone with a matching talent you need!

But how does everyone stay informed about these scores? This is where the **`attention weight matrix`** comes in! It's like a giant scoreboard displayed at the party, where each cell shows the attention score between a specific word pair. For example, the cell at row "Right" and column "Shift" would hold the score indicating how well these words relate.

But how does this translate into updating the word embeddings?
That's where **softmax** steps in, acting as the **regulator** who assigns weights to each score based on its relative importance.

Imagine the **regulator** listening to each word's scores and saying, "Okay, so 'Shift' is quite relevant for determining 'Right' meaning in this context, so you get a high weight."

Here's how softmax works its magic:

1. **Listen to All Scores:** It takes all the attention scores for a specific word (a row in the our `attention weight matrix`) as input.

2. **Apply the Formula:** It uses a mathematical formula to consider the relative strength of each score compared to the others. This ensures that scores that are much higher than others have a larger impact on the final weights.

3. **Distribute the Weights:** Softmax transforms these relative strengths into weights between 0 and 1, ensuring everyone gets a fair share of attention but the most relevant ones get a bigger slice.

Now, each word has a set of weighted values based on its interactions with others. Think of it as collecting insights from the most relevant conversations at the party. These weights are then used to perform a weighted combination of everyones values in order to create a richer, context-aware representation of the word's meaning.

In our example, "Right" in "Do what is Right" might learn a stronger "morality" value, while in "Shift to your Right," it gains a stronger "direction" value.

**The result? Word embeddings that truly reflect their meaning in each sentence, just like people adapting their communication based on the context!**

**Bonus Math (optional):**

The core calculation behind SDPA involves the attention weight matrix and softmax, expressed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{{Q.K^T}}{{\sqrt{d_k}}}\right) .V$$

where:

- $Q$, $K$, $V$ represent the query, key, and value vectors for every word joined as rows of the corresponding matrix.
> *Note: $K^T$ means that the matrix K was transposed, we do this to align the vectors dimensions apples to apples 🍏 before the similarity check.*

- **$\cdot$** denotes the dot product (measuring similarity).

- $\sqrt{d_k}$ is a scaling factor for stability (so very high scores don't totally drown out lower ones).

- $\text{softmax}$ distributes weights between 0 and 1 based on attention scores.

- **$\cdot$** $V$ uses the **weighted attention scores** to do a weighted combination of everyones values to get a new richer embedding.


#### **Multi-Head Attention**
This is a mechanism in the Self-Attention process where multiple "heads" or focus groups are used. Remember our word party? Multi-head attention throws another twist! Imagine multiple groups (heads) at the party, each focusing on different aspects of the conversation. For instance, one head might concentrate on semantic context, while another might focus on emotional context.

**Here are the details:**

* Each group (head) has its own "attention weight matrix" showing how relevant other words are to their focus.
* They analyze independently, like different games happening at once.
* In the end, they concatenate (Concat in the image above 👆👆) their insights, creating a richer understanding of each word, like piecing together clues from different groups.

#### **Masked Multi-Head Attention**
This is used in the decoder layer to prevent the model from seeing future words. This is achieved by replacing entries above the main diagonal of the attention matrix with `-inf` before performing softmax, a technique known as **Causal Masking**. (Imagine it as putting black tape above the main diagonal of the attention weight matrix).

For example, in a sentence "The cat sat on the mat.", for the word "sat", Masked Multi-Head Attention only considers "The" and "cat", ignoring "on", "the", and "mat".

In addition to causal masking, a **Padding Mask** is used to prevent the model from attending to `[PAD]` tokens added to equalize the lengths of sequences in a batch. The attention scores of `[PAD]` tokens are set to `-inf`, ensuring these tokens do not affect the final attention output.

In the case of padding mask, if we have a batch of two sequences: ["The cat sat down", "Good Morning [PAD] [PAD]"], the model's focus remains solely on the meaningful words in the sequence.

Note: Multi-Head & Masked Multi-Head Attention also have a projection layer (The Linear in the above 👆👆 image). Its job is to project these embeddings, updated from word context, to a more concise form for the Model.

💡Note: In our implementation below instead of creating multiple heads each with their Networks to get queries (Q), keys (K) and values (V), we are going to cleverly use a big network each for getting Q, K, and V and we will share the output of this networks to all the heads (This bascially does the same thing as with creating multiple heads, this is just a more compute effective way).

#### Encoeder Block
We crossed out the **Add&Norm** because we are replacing it with **ReZero** in this model (as we will see shortly).

The first layer shown is a **Multi-Head Attention** layer within the encoder block. This is called a **Self-Attention** layer because the multi-head attention mechanism draws its queries, keys and values only from the encoder input embeddings themselves.

In other words, the input embeddings are enriched solely based on relationships within themselves, without any external context.

We also notice the **skip connection** arrow that bypasses this Self-Attention layer, connecting the input directly to the output. This allows signals to propagate easily through the model. With ReZero, we omit the Layer normalization in the Add&Norm, instead doing a weighted addition as follows:

$$\text{skip} + \text{re_zero_weight} \times \text{output}$$

There is also a **dropout** layer (not shown in the image) that randomly sets some outputs to zero with a certain probability. This forces the model to utilize as much information as it can get. So the ReZero connection actually looks like:

$$\text{skip} + \text{re_zero_weight} \times \text{dropout}(\text{output})$$

Next is the **feed forward network**, whose purpose is to process all the information extracted by the previous layers into a more organized and understandable representation for later stages of the model. It also employs a residual skip connection.





