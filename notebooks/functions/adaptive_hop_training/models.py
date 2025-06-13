import random
import numpy as np
import torch

from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    AutoModel,
    RobertaTokenizer,
    RobertaForSequenceClassification
)

class QueryGenerator:
    def __init__(self, device):
        """
        Initialize the QueryGenerator with a device (CPU or GPU).
        Loads the T5-Flan model and tokenizer for generating queries.
        """
        self.device = device
        self.query_tokenizer, self.query_generator = self.get_query_model(device)


    @staticmethod
    def get_query_model(device):
        """Initialize and return T5-Flan query generator model and tokenizer."""
        QUERY_MODEL_NAME = "google/flan-t5-small"

        query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        query_generator = AutoModelForSeq2SeqLM.from_pretrained(QUERY_MODEL_NAME).to(device)
        query_generator.eval()
        
        return query_tokenizer, query_generator

    def generate_query(self, question, gen_config, context="", use_fewshot=True):
        """Generate a search query using T5-Flan
        
        Args:
            question: The question to generate a query for
            context: Optional context to inform query generation
            use_fewshot: Whether to use few-shot examples
            
        Returns:
            Generated query string
        """
        # Construct base prompt
        if context:
            prompt = f"Context: {context}\n\nGenerate a search query for: {question}"
        else:
            prompt = f"Generate a search query for: {question}"

        # Add few-shot examples if available and requested
        if use_fewshot and fewshot_examples:
            prompt = "\n\n".join(fewshot_examples) + "\n\n" + prompt

        # Tokenize and generate
        inputs = query_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=gen_config['max_input_length'],
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = query_generator.generate(
                **inputs,
                max_new_tokens=gen_config['max_new_tokens'],
                do_sample=gen_config['do_sample'],
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                num_return_sequences=1
            )
        
        query = query_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return query
    
    
def get_colbert_model(device):
    """Initialize and return ColBERT model and tokenizer for retrieval scoring."""

    COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"
    
    colbert_tokenizer = AutoTokenizer.from_pretrained(COLBERT_MODEL_NAME)
    colbert_model = AutoModel.from_pretrained(COLBERT_MODEL_NAME).to(device)
    colbert_model.eval()
    
    return colbert_tokenizer, colbert_model

def get_classifier_model(device):
    """Initialize and return RoBERTa relevance classifier model and tokenizer."""
    
    CLASSIFIER_MODEL_NAME = "roberta-base"
    
    classifier_tokenizer = RobertaTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)
    classifier_model = RobertaForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME, num_labels=2
    ).to(device)
    classifier_model.eval()
    
    return classifier_tokenizer, classifier_model


