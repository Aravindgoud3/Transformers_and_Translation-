# Transformer Architecture & implementation of Machine Translation 

## Overview

This project demonstrates a Transformer-based Machine Translation system built after studying and understanding the Transformer architecture as explained in The Illustrated Transformer and the paper “Attention Is All You Need”.

The goal of this project is twofold:

1. To understand and explain the internal architecture of Transformers (encoder–decoder, attention, positional encoding, etc.)
2. To apply that understanding by implementing a working machine translation pipeline using pretrained Transformer models, evaluation metrics, and a user interface.

---

## Understanding the Transformer Architecture

### High-Level Architecture
![Transformers](img/t.png)

The Transformer is an encoder–decoder architecture designed for sequence-to-sequence tasks such as machine translation.

* The Encoder reads and understands the input sentence.
* The Decoder generates the translated output sentence step by step.
* Unlike RNNs, Transformers process tokens in parallel, making training faster and more scalable.

---

### Encoder

The encoder consists of a stack of identical encoder layers. Each encoder layer contains:

1. Self-Attention Layer
   Allows each word in the input sentence to attend to all other words and capture contextual meaning.

2. Feed-Forward Neural Network (FFN)
   A fully connected network applied independently to each position.

Each sub-layer is followed by:

* Residual connections
* Layer normalization

This helps stabilize training and improve gradient flow.

---

### Self-Attention Mechanism

Self-attention enables the model to understand relationships between words in a sentence.

For each input word, three vectors are created:

* Query (Q)
* Key (K)
* Value (V)

Steps involved:

1. Compute attention scores using dot product of Query and Key
2. Scale the scores and apply softmax
3. Weight the Value vectors using these scores
4. Sum the weighted values to produce the attention output

This allows the model to resolve ambiguities (e.g., pronouns like "it") by attending to relevant words.

---

### Multi-Head Attention

Instead of a single attention operation, Transformers use multiple attention heads.

Benefits:

* Each head focuses on different linguistic aspects (syntax, semantics, long-range dependencies)
* Improves representation learning

Outputs from all heads are concatenated and projected through a linear layer.

---

### Positional Encoding

Since Transformers do not process tokens sequentially, positional encoding is added to input embeddings.

* Uses sinusoidal functions (sine and cosine)
* Encodes word order and relative positions
* Enables the model to generalize to longer sequences

---

### Decoder

The decoder is also a stack of layers, each containing:

1. Masked Self-Attention
   Prevents the model from seeing future tokens during generation.

2. Encoder–Decoder Attention
   Allows the decoder to focus on relevant parts of the input sentence.

3. Feed-Forward Neural Network

Like the encoder, residual connections and layer normalization are applied.

---

### Output Layer

* The decoder outputs a vector representation
* A linear layer maps it to vocabulary size
* Softmax converts it into probabilities
* The token with the highest probability is selected as output

This process repeats until an end-of-sequence token is generated.

---

## Project Implementation: Machine Translation

### Task Description

In this project, I implemented a Transformer-based machine translation system that translates English text into multiple target languages using pretrained encoder–decoder Transformer models (MarianMT).

The implementation reflects a practical application of the Transformer concepts studied from *The Illustrated Transformer*, including encoder–decoder attention, masked decoding, and token-based generation.

---

### Code-Level Implementation Details

The project is implemented in Python using PyTorch, Hugging Face Transformers, and Gradio.

#### Model Selection

* Used pretrained MarianMT models from the Helsinki-NLP group:

  * English → Hindi (opus-mt-en-hi)
  * English → French (opus-mt-en-fr)
  * English → Spanish (opus-mt-en-es)
* These models are based on the Transformer encoder–decoder architecture.

#### Device Handling

* Automatically detects and uses GPU (CUDA) if available, otherwise falls back to CPU.
* Ensures efficient inference during translation.

#### Tokenization and Encoding

* Used AutoTokenizer to:

  * Convert input text into token IDs
  * Handle padding and truncation
  * Prepare tensors compatible with Transformer models

#### Translation Pipeline

* Implemented a reusable translate() function that:

  * Accepts input text and target language
  * Encodes text into tensors
  * Uses the Transformer decoder to generate translated tokens
  * Decodes tokens back into readable text

#### Text Generation

* Used model.generate() for sequence generation
* Limited output length to avoid excessively long translations
* Applied greedy decoding suitable for demo and evaluation purposes

#### Evaluation Using BLEU Score

* Integrated sacreBLEU for optional evaluation
* Computes sentence-level BLEU score when a reference translation is provided
* Helps measure translation quality objectively

#### Interactive User Interface

* Built an interactive web UI using Gradio
* Features:

  * Input textbox for English text
  * Dropdown to select target language
  * Optional reference translation input
  * Real-time translation output with BLEU score

---

### Key Components of the Implementation

* Pretrained Transformer Models
  Used encoder–decoder Transformer models for translation tasks.

* Tokenization
  Input text is tokenized and converted into model-compatible tensors.

* Translation Pipeline
  A reusable translation function that:

  * Accepts input text
  * Generates translated output
  * Supports GPU/CPU execution

* Evaluation
  Translation quality is evaluated using BLEU score via sacrebleu.

* Interactive User Interface
  Built using Gradio, allowing users to:

  * Enter text
  * Select translation direction
  * View translated output in real time

---

## What This Project Demonstrates

* Strong understanding of Transformer architecture
* Practical application of self-attention and encoder–decoder models
* Ability to move from theory to implementation
* Experience with NLP pipelines, evaluation metrics, and model deployment

---

## Summary

Implemented a Transformer-based machine translation system by applying core concepts such as self-attention, multi-head attention, positional encoding, and masked decoding, with interactive UI and BLEU score evaluation.

---

## Future Enhancements

* Fine-tuning models on custom datasets (e.g., English–Telugu)
* Adding beam search for better decoding
* Deploying the application as a web service
* Experimenting with modern positional encodings (RoPE)

---

## References

* Attention Is All You Need (Vaswani et al.)
* The Illustrated Transformer by Jay Alammar
* Hugging Face Transformers Documentation
