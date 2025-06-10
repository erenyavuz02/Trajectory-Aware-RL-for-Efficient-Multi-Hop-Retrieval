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
test_dataset = dataset['test']

def build_fewshot_prompt(question, context="", add_fewshot=False):
    num_fewshots = random.randint(1, 3)
    fewshots = random.sample(FEWSHOT_EXAMPLES, num_fewshots)

    fewshot_str = "Examples:\n"
    for ex in fewshots:
        fewshot_str += f"Question:{ex['question']}\nQuery:{ex['query']}\n\n"

    context_str = f"Context:\n{context}\n\n" if context else ""

    task_str = f"Generate a search query for the following question:\n{question}"

    return f"{fewshot_str}{context_str}{task_str}"


# === Embedding utility ===
def compute_colbert_embeddings(texts):
    encoded = colbert_tokenizer(
        texts,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        output = colbert_model(**encoded).last_hidden_state
    masks = encoded["attention_mask"].bool()
    return [output[i][masks[i]].cpu().numpy() for i in range(len(texts))]

# === Scoring utility ===
def maxsim_score(query_emb, doc_embs):
    return float((torch.matmul(query_emb, doc_embs.T)).max(dim=1).values.sum())

def compute_ap_recall_precision(supporting_pairs, retrieved_ids, sentence_metadata):
    if not retrieved_ids or not supporting_pairs:
        return 0.0, 0.0, 0.0
        
    retrieved_pairs = {
        (sentence_metadata[i]["title"], sentence_metadata[i]["sent_idx"]) for i in retrieved_ids
    }
    hits = [1 if (sentence_metadata[i]["title"], sentence_metadata[i]["sent_idx"]) in supporting_pairs else 0 for i in retrieved_ids]
    
    # Calculate AP (Average Precision)
    ap = sum(hits[i] / (i + 1) for i in range(len(hits)) if hits[i]) / max(sum(hits), 1)
    
    # Calculate regular precision
    precision = sum(hits) / len(retrieved_ids) if retrieved_ids else 0
    
    # Calculate recall
    recall = sum(hits) / len(supporting_pairs) if supporting_pairs else 0
    
    return ap, precision, recall

def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)

def evaluate_hotpotqa(
    eval_dataset,
    query_generator,
    query_tokenizer,
    num_hops=2,
    top_k_retrieval=5,
    max_new_tokens=20 # Allow more tokens for potentially longer queries
):
    print(f"Starting evaluation with {len(eval_dataset)} samples...")

    # Initialize per-hop metrics
    metrics_per_hop = [{
        "total_ap": 0.0,
        "total_precision": 0.0,
        "total_recall": 0.0,
        "num_samples": 0
    } for _ in range(num_hops)]

    all_results = [] # To store detailed results for inspection
    for idx in tqdm(range(10), desc="Evaluating Samples"):
        sample = eval_dataset[idx]
        print(f"Evaluating sample {sample}")
        question = sample['question']
        supporting_facts = sample['supporting_facts']

        # Flatten context
        context_titles = sample['context']['title']
        context_sentences_grouped = sample['context']['sentences']
        flattened_sentences = []
        sentence_metadata = []
        for title, sentences in zip(context_titles, context_sentences_grouped):
            for i, sent in enumerate(sentences):
                flattened_sentences.append(sent)
                sentence_metadata.append({"title": title, "sent_idx": i})

        # Compute embeddings for the entire context once
        context_embeddings = compute_colbert_embeddings(flattened_sentences)

        # Convert list of numpy arrays to a list of tensors for maxsim_score
        # vector_store_embeddings_for_scoring stores individual document token embeddings
        vector_store_embeddings_for_scoring = [torch.tensor(emb, dtype=torch.float32).to(device) for emb in context_embeddings]


        current_context = ""  # No context for the first hop

        # Ground truth supporting pairs for the current question
        ground_truth_supporting_pairs = set(zip(supporting_facts['title'], supporting_facts['sent_id']))

        # Store results for this question
        question_results = {
            "question": question,
            "ground_truth_supporting_pairs": list(ground_truth_supporting_pairs),
            "hops": []
        }

        # Skip questions with no supporting facts, as AP/Recall/F1 are ill-defined
        if not ground_truth_supporting_pairs:
            # print(f"Skipping question '{question}' due to no supporting facts.") # Keep this for debugging if needed
            continue

        for hop in range(num_hops):

            prompt = build_fewshot_prompt(question, context=current_context)

            inputs = query_tokenizer(
                prompt,
                return_tensors="pt",
                padding=True, # Apply padding if batching (though num_return_sequences=1 here)
                truncation=True
            ).to(query_generator.device)

            # For T5 (Seq2Seq), you don't typically slice by prompt_length from `outputs.sequences`
            # Instead, the decoder output is directly the generated text.
            # You feed `input_ids` to the encoder, and the decoder generates.
            outputs = query_generator.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False, # You can set this to False for deterministic generation
                top_p=0.9,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=query_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False
            )

            # For T5 (Seq2Seq models), the `generated_sequences` are just the decoded output.
            # No need to slice by `prompt_length`.
            generated_sequence = outputs.sequences[0]
            generated_query = query_tokenizer.decode(generated_sequence, skip_special_tokens=True).strip()

            if not generated_query:
                print(f"Warning: Empty query generated for question: '{question}' hop: {hop}.")
                continue # Skip if empty query

            # === Retrieval and Scoring ===
            query_emb_list = compute_colbert_embeddings([generated_query])
            if not query_emb_list:
                print(f"Warning: No embedding generated for query: '{generated_query}' for question: '{question}' hop: {hop}.")
                continue # Skip if embedding fails

            query_emb = query_emb_list[0] # This is already a numpy array from compute_colbert_embeddings
            # No need for `torch.tensor().to(device)` here, as `maxsim_score` will handle it for each call
            # `maxsim_score` itself converts to tensor on the device.

            scores = []
            for doc_emb in vector_store_embeddings_for_scoring: # Iterate through each document's token embeddings
                scores.append(maxsim_score(query_emb, doc_emb)) # query_emb (numpy), doc_emb (tensor already)

            if not scores:
                continue

            # Get top_k retrieved document indices
            top_indices = np.argsort(scores)[-top_k_retrieval:][::-1]

            # Calculate AP, precision, and recall for the current hop
            ap, precision, recall = compute_ap_recall_precision(
                ground_truth_supporting_pairs, 
                top_indices, 
                sentence_metadata
            )

            f1 = calculate_f1(precision, recall)

            # Accumulate scores per hop
            metrics_per_hop[hop]["total_ap"] += ap
            metrics_per_hop[hop]["total_precision"] += precision
            metrics_per_hop[hop]["total_recall"] += recall
            metrics_per_hop[hop]["num_samples"] += 1

            retrieved_context = [flattened_sentences[i] for i in top_indices]

            question_results["hops"].append({
                "hop": hop,
                "generated_query": generated_query,
                "raw_generated_query": generated_query,
                "ap": ap,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "top_k_retrieved_docs": retrieved_context,
                "top_k_retrieved_ids": top_indices.tolist()
            })

            # Update current_context for the next hop
            current_context = "\n".join(retrieved_context)

        all_results.append(question_results)

    # Calculate and print metrics for each hop
    print("\n=== Per-Hop Evaluation Summary ===")
    hop_summaries = []
    for hop in range(num_hops):
        num_samples = metrics_per_hop[hop]["num_samples"]
        if num_samples > 0:
            avg_ap = metrics_per_hop[hop]["total_ap"] / num_samples
            avg_precision = metrics_per_hop[hop]["total_precision"] / num_samples
            avg_recall = metrics_per_hop[hop]["total_recall"] / num_samples
            avg_f1 = calculate_f1(avg_precision, avg_recall)
            
            print(f"\nHop {hop + 1} Metrics:")
            print(f"Number of Samples: {num_samples}")
            print(f"Average AP: {avg_ap:.4f}")
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"Average Recall: {avg_recall:.4f}")
            print(f"Average F1: {avg_f1:.4f}")
            
            hop_summaries.append({
                "hop": hop + 1,
                "num_samples": num_samples,
                "average_ap": avg_ap,
                "average_precision": avg_precision,
                "average_recall": avg_recall,
                "average_f1": avg_f1
            })

    # Calculate overall metrics (averaged across all hops)
    total_samples = sum(hop["num_samples"] for hop in metrics_per_hop)
    overall_ap = sum(hop["total_ap"] for hop in metrics_per_hop) / total_samples if total_samples > 0 else 0.0
    overall_precision = sum(hop["total_precision"] for hop in metrics_per_hop) / total_samples if total_samples > 0 else 0.0
    overall_recall = sum(hop["total_recall"] for hop in metrics_per_hop) / total_samples if total_samples > 0 else 0.0
    overall_f1 = calculate_f1(overall_precision, overall_recall)

    print("\n=== Overall Metrics (Averaged Across Hops) ===")
    print(f"Total Samples Evaluated: {total_samples}")
    print(f"Overall AP: {overall_ap:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1: {overall_f1:.4f}")

    return {
        "overall_metrics": {
            "average_ap": overall_ap,
            "average_precision": overall_precision,
            "average_recall": overall_recall,
            "average_f1": overall_f1,
            "total_samples": total_samples
        },
        "per_hop_metrics": hop_summaries,
        "detailed_results": all_results
    }
    
    
from transformers import AutoModelForSeq2SeqLM

# Model to evaluate
model_path= "google/flan-t5-small"
model_to_eval = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
model_to_eval_tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test dataset
dataset = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)
eval_dataset = dataset['test']

# ColBERT
colbert_tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
colbert_model = AutoModel.from_pretrained("colbert-ir/colbertv2.0").to(device)
colbert_model.eval()


# --- Run Evaluation ---
evaluation_metrics = evaluate_hotpotqa(
    eval_dataset=eval_dataset,
    query_generator=model_to_eval,
    query_tokenizer=model_to_eval_tokenizer,
    num_hops=2,           # Keep consistent with your training/preference dataset generation
    top_k_retrieval=5,    # Keep consistent with your preference dataset generation
    max_new_tokens=20     # Adjust as needed for query length
)

# --- Save Results (Optional) ---
output_filename = "hotpotqa_evaluation_results.json"
with open(output_filename, "w") as f:
    json.dump(evaluation_metrics, f, indent=4)
print(f"\nDetailed evaluation results saved to {output_filename}")