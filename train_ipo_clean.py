import torch
import torch.nn.utils as utils
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from tqdm import tqdm
import wandb
import os

class PreferenceDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        self.data = []
        for question, entry in raw_data.items():
            for hop, hop_data in entry["hops"].items():
                queries = hop_data["queries"]
                preferences = hop_data["preference_pairs"]
                for i, j in preferences:
                    self.data.append({
                        "question": question,
                        "preferred": queries[i],
                        "dispreferred": queries[j]
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Replace the model loading cell with:

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True
)

model.train()
print(f"Model {model_name} loaded successfully!")

os.environ["WANDB_API_KEY"] = "57b8585a9cdb363d54a7d215dd95c824d880868b"
wandb.login()

def compute_logp(prompt, completion):
    """Compute log probability of completion given prompt for seq2seq model"""
    # For T5, input is the prompt, target is the completion
    input_encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    target_encoded = tokenizer(completion, return_tensors="pt", truncation=True, max_length=64)

    device = next(model.parameters()).device
    input_ids = input_encoded.input_ids.to(device)
    input_attention_mask = input_encoded.attention_mask.to(device)
    labels = target_encoded.input_ids.to(device)

    # Replace pad tokens in labels with -100
    labels[labels == tokenizer.pad_token_id] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=input_attention_mask,
        labels=labels
    )
    return -outputs.loss

def ipo_loss(logp_win, logp_lose, tau=0.05):
    """Compute IPO loss"""
    diff = logp_win - logp_lose - 0.5 / tau
    return (diff ** 2).mean()

# Training setup
parent_path = ''
#dataset_path = f'{parent_path}/preference_dataset_hotpotqa_final.json'
dataset_path = 'preference_dataset_hotpotqa_final.json'
dataset = PreferenceDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = AdamW(model.parameters(), lr=5e-6)
tau = 0.05
num_epochs = 3

print(f"Dataset size: {len(dataset)}")
print(f"Starting IPO training...")

# check if this direcotry exists epoch_save_path = "/trained model"
epoch_save_path = "trained_model"
os.makedirs(epoch_save_path, exist_ok=True)

wandb.init(
    project="c438_project",  # Name of the project in W&B
    name="ipo_training_run",    # A specific name for this run
    config={
        "learning_rate": optimizer.defaults['lr'],
        "num_epochs": num_epochs,
        "batch_size": dataloader.batch_size,
        "model_name": model.name_or_path,
        "tau": tau,
    }
)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        questions = batch["question"]
        preferred = batch["preferred"]
        dispreferred = batch["dispreferred"]

        logp_w_list = []
        logp_l_list = []

        for q, w, l in zip(questions, preferred, dispreferred):
            prompt = f"Generate a search query for: {q}\nQuery: "

            logp_w = compute_logp(prompt, w.strip())
            logp_l = compute_logp(prompt, l.strip())

            logp_w_list.append(logp_w)
            logp_l_list.append(logp_l)

        logp_w_batch = torch.stack(logp_w_list)
        logp_l_batch = torch.stack(logp_l_list)

        loss = ipo_loss(logp_w_batch, logp_l_batch, tau)

        if torch.isnan(loss):
            print(f"Skipping batch {batch_idx} due to NaN loss.")
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = loss.item()
        total_loss += current_loss
        avg_loss = total_loss / (batch_idx + 1)

        if (batch_idx + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")

        wandb.log({"step_loss": current_loss, "average_loss": avg_loss})

    epoch_avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch + 1}] Average Loss: {epoch_avg_loss:.4f}")

    wandb.log({"epoch": epoch + 1, "epoch_average_loss": epoch_avg_loss})

    model.save_pretrained(epoch_save_path)
    tokenizer.save_pretrained(epoch_save_path)
    print(f"Epoch {epoch + 1} model saved to {epoch_save_path}")

print("Training completed!")

# --- Finish the W&B run ---
wandb.finish()




