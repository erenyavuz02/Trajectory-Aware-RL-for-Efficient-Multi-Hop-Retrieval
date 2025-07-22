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


## How to Run files

### Preference Dataset Generation
preference_dataset_generation.ipynb

- Install the dependencies in the first cell of the notebook in the colab
- For different models for query generation, you can change the model name in the second cell of the notebook
- In the third cell, you can download the hotpotqa dataset from HuggingFace or use the one in the repository by uploading it to the colab.

  change the dataset path to get the splitted dataset from the repository by defining the directory of the dataset in the notebook
  ```python
  train_filename = "hotpot_train_5000samples_2025-06-11_18-31-48.jsonl"
  val_filename = "hotpot_val_1000samples_2025-06-11_18-31-48.jsonl"
  ```
- Run the finetuning cell if not wanted to use the default dataset.

### Training IPO Model
train_ipo_clean.ipynb

- Install the dependencies in the first cell of the notebook in the colab
- This notebook creates a torch Dataset class using the preference dataset created in the previous step. This used the non formatted preference dataset created in the folder
  `new_datasets/preference_datasets/`
- The training is done using the T5-Flan model with the preference dataset.
- before training the model change the name of the saved folder for better convention
  ```python
  model_name = "epoch_x_{model_configuration}"
  ```
  where `x` is the epoch number and `model_configuration` is the configuration of the model used for training.

### Evaluating Models
evaluate_models.ipynb

- Install the dependencies in the first cell of the notebook in the colab
- This notebook evaluates the trained models using the HotpotQA dataset and the ColBERT embeddings
- The evaluation is done using the few-shot examples created in the previous step.
- The few-shot examples are loaded from the `fewshot_examples.json` file in the `datasets/` folder.
- The evaluation is done using the T5-Flan model with the preference dataset.
- The evaluation results are saved in the `evaluation_results/` folder.
