# 🧠 NLP Text Reconstruction & Semantic Similarity Project

This repository contains a comprehensive NLP pipeline developed for reconstructing grammatically and semantically unclear English text using multiple strategies. The project integrates traditional rule-based methods, spaCy-based preprocessing, and Transformer models (T5), and evaluates the output using advanced similarity metrics and visualizations.

---

## 📋 Project Overview

The objective is to reconstruct text with unclear grammar/structure into clearer, more grammatically correct English using:
- Custom rule-based corrections (`auto_reconstruct`)
- Syntactic analysis with **spaCy**
- Fine-tuned **T5 Transformers**
- Combined hybrid approaches

Reconstruction quality is assessed using **BERT-based cosine similarity**, **Word2Vec**, and **lexical overlap** metrics. Additionally, **embedding visualizations** are produced with **PCA/t-SNE** for interpretability.

---

## 🛠️ Technologies & Tools

- **Python 3.10**
- **Transformers (Hugging Face)** – T5 & BERT
- **spaCy** – syntactic segmentation & token filtering
- **Gensim** – Word2Vec similarity comparison
- **scikit-learn** – cosine similarity, PCA, t-SNE
- **Matplotlib** – embedding visualization

---

## 🧪 Features

- 🔁 **Multiple Text Reconstruction Pipelines**:  
  - `auto_reconstruct()` — rule-based, handcrafted.
  - `spacy_reconstruct()` — token-level sentence adjustment.
  - `transformers_reconstruct()` — paraphrasing via T5 model.
  - `spacy_auto_reconstruct()` — hybrid of rule-based and syntactic approaches.

- 📊 **Semantic Evaluation**:
  - Cosine similarity using **BERT embeddings**
  - **Lexical overlap (Jaccard index)** via Word2Vec token embeddings

- 📈 **Embedding Visualization**:
  - PCA for initial reduction
  - t-SNE for 2D plotting of semantic spaces before and after reconstruction

---

## 📁 File Structure

├── main.py # Entry point, manages the full pipeline
├── reconstruction.py # Contains all reconstruction methods
├── similarity.py # Similarity metrics & visualizations
└── README.md # Project documentation

---

## 📚 Additional Useful Resources

- *Speech and Language Processing* – Jurafsky & Martin
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- Devlin et al. (2019) – BERT: Pre-training of Deep Bidirectional Transformers
- Vaswani et al. (2017) – Attention Is All You Need

---
Clone the repository:
