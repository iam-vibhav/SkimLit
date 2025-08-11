# SkimLit Remake: Classifying Research Paper Abstracts

This project is a remake of the [Pubmed 20k RCT](https://arxiv.org/pdf/1710.06071) paper, which introduces models for classifying sentences in medical abstracts into structured categories such as **BACKGROUND**, **OBJECTIVE**, **METHODS**, **RESULTS**, and **CONCLUSIONS**.  

The goal is to automatically label sentences from research abstracts, enabling faster literature review and structured data extraction for downstream applications.

---

## 📌 Project Overview

- **Task:** Classify each sentence of a research paper abstract into one of 5 categories.  
- **Dataset:** Based on PubMed abstracts (processed into SkimLit format).  
- **Approach:**  
  1. Start with simple baselines (Naive Bayes + TF-IDF).  
  2. Progressively build more advanced neural architectures:
     - Token-level CNN
     - Universal Sentence Encoder (USE)  
     - Character-level embeddings (Conv1D + BiLSTM)  
     - Hybrid token + char embeddings  
     - **Tribrid Model (model_5)**: Pretrained token embeddings + character embeddings + positional embeddings (line number and total lines).  

---

## 📂 Models & Results

Validation metrics across models (evaluated on 10% of the dataset for efficiency):

| Model                                   | Accuracy | F1 Score |
|-----------------------------------------|----------|----------|
| **Naive Bayes + TF-IDF**                | 0.72     | 0.70     |
| **Conv1D Token Embeddings**             | 0.76     | 0.75     |
| **Universal Sentence Encoder (USE)**    | 0.78     | 0.77     |
| **Conv1D Char Embeddings**              | 0.75     | 0.73     |
| **Token + Char Embeddings**             | 0.80     | 0.79     |
| **Token + Char + Positional (model_5)** | **0.85** | **0.84** |

> ✅ **model_5** outperformed all previous baselines and achieved strong results despite being trained on only 10% of the dataset.

---

## ⚙️ Architecture of `model_5`

The final **Tribrid Model** combines multiple modalities:

- **Token embeddings:** Universal Sentence Encoder (trainable)  
- **Character embeddings:** BiLSTM over char sequences  
- **Positional features:** Line number + total lines one-hot encoded  
- **Fusion layer:** Concatenation → Dense (256) → Dropout  
- **Output:** Softmax over 5 abstract categories  

---

## 🚀 Training Setup

- **Framework:** TensorFlow / Keras  
- **Optimizer:** Adam  
- **Loss:** Categorical Crossentropy (with label smoothing = 0.1)  
- **Batch Size:** 32  
- **Epochs:** 5 (with checkpoints for memory efficiency)  
- **Evaluation Metrics:** Accuracy & Weighted F1 Score  

---

## 📊 Experimental Notes

- All models prior to `model_5` were cleared from memory to avoid RAM overflow, while metrics were stored for later comparison.  
- The tribrid architecture significantly boosted performance by incorporating **positional context** in addition to token and char embeddings.  
- Although trained on 10% of the dataset due to memory constraints, results were competitive with the original SkimLit paper (which used the full dataset).  

---

## 🧪 Using the Wrapper

The project also includes a wrapper to run predictions on **raw research paper text**.  
You can simply pass a raw abstract, and the pipeline will preprocess the text, vectorize inputs, and return predictions for each line.

```python
raw_text = """The dominant sequence transduction models are based on complex recurrent...
...best models from the literature."""
results = predict_research_paper(raw_text)

for r in results:
    print(f"{r['prediction']}: {r['text']}")
