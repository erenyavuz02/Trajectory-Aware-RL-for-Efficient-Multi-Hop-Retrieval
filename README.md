# Trajectory-Aware-RL-for-Efficient-Multi-Hop-Retrieval

## Project Overview

This project implements a trajectory-aware reinforcement learning approach for efficient multi-hop question answering and retrieval using the HotpotQA dataset. The system combines preference learning, adaptive hop classification, and query generation for improved multi-step reasoning in question answering tasks.

## Files and Folders Description

**Main Scripts:**

- `evaluate_hotpotqa.py` - Evaluation script for HotpotQA dataset using ColBERT embeddings and T5-based query generation with few-shot prompting
- `train_ipo_clean.py` - Implementation of Identity Preference Optimization (IPO) training using T5-Flan model with preference datasets for query generation improvement

**Data Files:**

- `hotpot_train_5000samples_2025-06-11_18-31-48.jsonl` - Training subset of HotpotQA dataset with 5000 samples
- `hotpot_val_1000samples_2025-06-11_18-31-48.jsonl` - Validation subset of HotpotQA dataset with 1000 samples
- `train_ipo.ipynb` - Jupyter notebook version of the IPO training implementation (DEPRECIATED)

**Datasets Folder:**

- `datasets/fewshot_examples.json` - Few-shot examples for query generation prompting
- `datasets/hotpotqa_fewshot_examples_adapt_hop.json` - Adaptive hop-specific few-shot examples for HotpotQA (DEPRECIATED)
- `datasets/preference_dataset/` - Contains preference datasets for training including final HotpotQA preferences and T5-Flan generated preferences

**Model Artifacts:**

- `epoch_1_model/` - Saved model checkpoint after first epoch containing T5 model weights, tokenizer configurations, and generation settings
- `pkl_files/` - Training checkpoints and intermediate results including model states at different training steps and adaptive preference datasets

**Notebooks:**


- `notebooks/evaluate_models.ipynb` - Model evaluation and performance analysis notebook
- `notebooks/hotpotqa_fewshot_generation.ipynb` - Few-shot example generation for HotpotQA dataset
- `notebooks/preference_dataset_generation.ipynb` - Preference dataset creation and processing
- `notebooks/train_ipo_clean.ipynb` - Clean version of IPO training notebook

- `notebooks/adaptive_hop_training.ipynb` - Implementation of adaptive hop classification training (CANCELLED)
- `notebooks/functions/adaptive_hop_training/` - Modular functions for adaptive hop training including classifier, dataset handling, and model implementations

**Supporting Files:**

- `fewshot_examples/` - Additional few-shot examples for different components of the system including general and HotpotQA-specific examples