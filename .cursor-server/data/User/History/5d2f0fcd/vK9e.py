from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import os
import json
from tqdm import tqdm
from huggingface_hub import login
import numpy as np
from collections import defaultdict
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
colbert_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0").to(device)
colbert_model.eval()

# === Hugging Face auth ===
login("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")

# === Dataset ===
dataset = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)
train_dataset = dataset['train'][:25000]
# LeReT uses around 25K items from their dataset.

'''
query_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True)
query_tokenizer.pad_token = query_tokenizer.eos_token

query_generator = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
query_generator.eval()
'''


fewshot_ex_path = 'fewshot_examples/fewshot_examples.json'

with open(fewshot_ex_path, 'r') as f:
    FEWSHOT_EXAMPLES = json.load(f)

print(f"Loaded {len(FEWSHOT_EXAMPLES)} few-shot examples.")