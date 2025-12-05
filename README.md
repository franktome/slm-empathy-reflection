# Empathy Scoring Models for LLM Agents

**Small Language Models (SLMs) for empathy evaluation in conversational AI**

## Overview

This repository provides three PyTorch/HuggingFace models for predicting empathy in conversational AI responses:

1.  **Continuous empathy scores** (1–5 scale: regression task)
2.  **Emotion labels** (32 classes: classification task)

All models are built on a **RoBERTa-base encoder** and are trained on the **EmpatheticDialogues** dataset using a custom hybrid empathy labeling pipeline.

---

## Model Summaries

| Model | Key Features | Training Loss | Latency (ms) |
| :--- | :--- | :--- | :--- |
| **Baseline Multitask** | Single-stage multitask training. | $L = L_{reg} + \lambda_{emo} \cdot L_{emo}$ ($\lambda_{emo}=0.1$) | 5–7 |
| **Sequential Transfer Multitask (STM)** | Two-stage training: Emotion-only $\rightarrow$ Multitask. | $L = L_{reg} + \lambda_{emo} \cdot L_{emo}$ | 5–7 |
| **Multi-View Cascade (MVC)** | **Best Performance.** Two input views (context-only & full dialogue). Empathy prediction explicitly uses emotion logits. | Custom cascade loss | 18–21 |

### 1. Baseline Multitask Architecture
A standard approach with a shared RoBERTa encoder and two separate prediction heads.
* **Encoder:** RoBERTa $\rightarrow$ `[CLS]` token
* **Heads:** Emotion (Classification) and Empathy (Regression)

### 2. STM — Sequential Transfer Multitask
Leverages emotion pre-training to better initialize the encoder for the empathy task.
* **Stage 1:** Fine-tune on **Emotion-only** task.
* **Stage 2:** Fine-tune on **Multitask** (Emotion + Empathy).

### 3. MVC — Multi-View Cascade
This model trades slightly higher latency for significant accuracy improvement by leveraging multiple views of the dialogue and cascading predictions.
* **View 1:** Context-only input.
* **View 2:** Full Context + Response input.
* **Cascade:** Empathy prediction is conditioned on the output **emotion logits**.


---

## Performance Results

Results averaged over three random seeds (17, 39, 43).

### Main Performance

| Model | MAE ↓ (Empathy) | Spearman ↑ (Empathy) | Emotion Acc ↑ |
| :--- | :--- | :--- | :--- |
| Baseline | 0.3977 | 0.7396 | 0.4150 |
| STM | 0.4946 | 0.7077 | 0.3951 |
| **MVC ($\lambda=0.1$)** | **0.3338** | **0.7574** | 0.4148 |

> **Key Finding:** MVC achieves **~15.5% lower MAE** than the Baseline model, demonstrating superior continuous empathy scoring.

### Effect of $\lambda_{emo}$ (Multitask Weight for MVC)

| $\lambda_{emo}$ | MAE ↓ | Spearman ↑ | Acc ↑ |
| :--- | :--- | :--- | :--- |
| 0.30 | 0.4033 | 0.7091 | 0.4305 |
| **0.10** | **0.3404** | **0.7505** | **0.4529** |
| 0.05 | 0.3576 | 0.7222 | 0.2691 |

The best performance balance is achieved with $\lambda_{emo} = 0.10$.

---

## Training Setup

### Main Hyperparameters

| Setting | Baseline | STM | MVC |
| :--- | :--- | :--- | :--- |
| Train batch size | 16 | 16 | 8 |
| LR | 3e-5 | 2e-5 | 2e-5 |
| Epochs | 5 | 10 | 4 |
| Grad accum | 2 | 2 | 1 |

### Additional Settings
* **Optimizer:** AdamW
* **Max length:** 256 tokens
* **Mixed precision (FP16):** Enabled

---

## How to Run

### 1. Install Dependencies
```bash
pip install torch transformers datasets evaluate scipy
```

### 2. Run in Order

Follow the Jupyter notebooks sequentially to replicate the results and models:

* **`data_preprocessing.ipynb`**: Build the dataset, generate augmentation variants, and compute the hybrid empathy labels.
* **`a_baseline.ipynb`**: Train the **Baseline Multitask Model**.
* **`b_sequential_transfer_multitask.ipynb`**: Train the **STM Model**.
* **`c_multi_view_cascade.ipynb`**: Train the **MVC Model** (with GoEmotions pretraining).

---

## Citation

If you use any part of this work, please cite the following paper:
@inproceedings{yun2025empathy, title={Empathy Scoring for LLM Agents via a Multi-View Cascade Small Language Model}, author={Yun, H.}, booktitle={AI Final Project, School of Undergraduate Studies, DGIST}, year={2025} }
