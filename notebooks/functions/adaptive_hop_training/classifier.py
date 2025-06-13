import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datetime import datetime
CURRENT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class Classifier:
    def __init__(self, classifier_model, classifier_tokenizer, device):
        """
        Initialize the Classifier with a model and tokenizer.
        
        :param classifier_model: Pre-trained model for classification
        :param classifier_tokenizer: Tokenizer for the classification model
        """
        self.classifier_model = classifier_model
        self.classifier_tokenizer = classifier_tokenizer
        self.device = device
        # Training Configuration
        

    def is_context_relevant(self, question: str, context: str) -> str:
        """
        Return "yes" if context is relevant to question, else "no"
        """
        # build input pair
        text = f"question: {question}  context: {context}"
        inputs = self.classifier_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.classifier_model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        return "yes" if pred == 1 else "no"

    def train(self, training_config, train_data, val_data):
        
        self.CLASSIFIER_TRAINING_CONFIG = training_config
        # Test before training
        print("Testing classifier with example question and context:")
        test_question = "Which magazine was started first Arthur's Magazine or First for Women?"
        test_context = "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century."

        result = self.is_context_relevant(test_question, test_context)
        print(f"Test example:")
        print(f"Question: {test_question}")
        print(f"Context: {test_context}")
        print(f"Prediction: {result}")

        # Train the classifier
        print("\nStarting classifier training...")
        best_accuracy = self.train_relevance_classifier(
            train_data=train_data,
            val_data=val_data
        )
    
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        
        # Test the trained classifier with examples from validation dataset
        print("Testing classifier with examples from validation dataset:")
        print("=" * 60)

        # Get some examples from validation dataset
        for i in range(5):  # Show 5 examples
            question = val_data['question'][i]
            context_data = val_data['context'][i]
            supporting_facts = val_data['supporting_facts'][i]

            print(f"\nExample {i+1}:")
            print(f"Question: {question}")
            print(f"Answer: {val_data['answer'][i]}")
            
            # Get supporting fact pairs for this question
            supporting_pairs = set(zip(supporting_facts['title'], supporting_facts['sent_id']))
            
            # Test with one relevant and one irrelevant context
            context_titles = context_data['title']
            context_sentences = context_data['sentences']
            
            # Find a supporting sentence
            relevant_context = None
            irrelevant_context = None
            
            for title_idx, (title, sentences) in enumerate(zip(context_titles, context_sentences)):
                for sent_idx, sentence in enumerate(sentences):
                    is_supporting = (title, sent_idx) in supporting_pairs
                    if is_supporting and relevant_context is None:
                        relevant_context = sentence
                    elif not is_supporting and irrelevant_context is None:
                        irrelevant_context = sentence
            
            # Test both contexts
            if relevant_context:
                result = self.is_context_relevant(question, relevant_context)
                print(f"Relevant context: {relevant_context[:100]}...")
                print(f"Classifier prediction: {result}")
            
            if irrelevant_context:
                result = self.is_context_relevant(question, irrelevant_context)
                print(f"Irrelevant context: {irrelevant_context[:100]}...")
                print(f"Classifier prediction: {result}")
            
            print("-" * 40)

        # Original test example
        test_question = "Which magazine was started first Arthur's Magazine or First for Women?"
        test_context = "Arthur's Magazine (1844–1846) was an American literary periodical published in Philadelphia in the 19th century."
        
        result = self.is_context_relevant(test_question, test_context)
        print(f"\nTest example:")
        print(f"Question: {test_question}")
        print(f"Context: {test_context}")
        print(f"Prediction: {result}")

    def train_relevance_classifier(self, train_data, val_data):
        """Train the relevance classifier"""

        print("Preparing datasets...")
        train_dataset = RelevanceDataset(data = train_data,
            tokenizer = self.classifier_tokenizer,
            max_length = self.CLASSIFIER_TRAINING_CONFIG['max_length'],
            num_samples=self.CLASSIFIER_TRAINING_CONFIG['num_samples']
        )

        val_dataset = RelevanceDataset(
            data = val_data,
            tokenizer = self.classifier_tokenizer,
            max_length = self.CLASSIFIER_TRAINING_CONFIG['max_length'],
            num_samples=self.CLASSIFIER_TRAINING_CONFIG['num_samples']  // 10 
        )

        train_loader = train_dataset.get_data_loader(
            batch_size=self.CLASSIFIER_TRAINING_CONFIG['batch_size'], 
            shuffle=True
        )

        val_loader = val_dataset.get_data_loader(
            batch_size=self.CLASSIFIER_TRAINING_CONFIG['batch_size'], 
            shuffle=False
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.classifier_model.parameters(),
            lr=self.CLASSIFIER_TRAINING_CONFIG['learning_rate'],
            weight_decay=self.CLASSIFIER_TRAINING_CONFIG['weight_decay']
        )

        total_steps = len(train_loader) * self.CLASSIFIER_TRAINING_CONFIG['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.CLASSIFIER_TRAINING_CONFIG['warmup_steps'],
            num_training_steps=total_steps
        )
        
        # validate model before training
        print("Validating model before training...")
        val_acc = self.evaluate_classifier(val_loader)
        print(f"Initial validation accuracy: {val_acc:.4f}")
        
        
        # Training loop
        self.classifier_model.train()
        best_val_acc = 0.0

        for epoch in range(self.CLASSIFIER_TRAINING_CONFIG['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.CLASSIFIER_TRAINING_CONFIG['num_epochs']}")

            total_loss = 0
            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.classifier_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.classifier_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Average training loss: {avg_loss:.4f}")
            
            # Validation
            val_acc = self.evaluate_classifier(val_loader)
            print(f"Validation accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.CLASSIFIER_TRAINING_CONFIG['save_model']:
                    model_save_path = f"relevance_classifier_best_{CURRENT_TIMESTAMP}.pt"
                    torch.save(self.classifier_model.state_dict(), model_save_path)
                    print(f"Best model saved: {model_save_path}")
        
        return best_val_acc

    def evaluate_classifier(self, data_loader):
        """Evaluate classifier performance"""
        self.classifier_model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.classifier_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        self.classifier_model.train()
        accuracy = accuracy_score(true_labels, predictions)
        
        return accuracy


class RelevanceDataset(Dataset):
    """Dataset for training relevance classifier"""

    def __init__(self, data, tokenizer, max_length=512, num_samples=1000):

        self.data = data

        # Extract questions, contexts, and labels
        questions, contexts, labels = self.create_training_data(num_samples=num_samples)

        self.questions = questions
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        label = self.labels[idx]
        
        # Format input as "question: [QUESTION] context: [CONTEXT]"
        text = f"question: {question} context: {context}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        
    def get_data_loader(self, batch_size=16, shuffle=True):
        """Create DataLoader for the dataset"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def create_training_data(self, num_samples=1000):
        """Create training data from HotpotQA dataset with balanced positive and negative examples"""
        import random
        
        positive_questions = []
        positive_contexts = []
        negative_questions = []
        negative_contexts = []
        
        print(f"Creating training data from {num_samples} samples...")

        for i in tqdm(range(min(num_samples, len(self.data['question'])))):
            question = self.data['question'][i]
            supporting_facts = self.data['supporting_facts'][i]
            context_data = self.data['context'][i]

            # Get supporting fact pairs
            supporting_pairs = set(zip(supporting_facts['title'], supporting_facts['sent_id']))
            
            # Collect all sentences with their labels
            context_titles = context_data['title']
            context_sentences = context_data['sentences']
            
            question_positives = []
            question_negatives = []
            
            for title_idx, (title, sentences) in enumerate(zip(context_titles, context_sentences)):
                for sent_idx, sentence in enumerate(sentences):
                    # Check if this sentence is a supporting fact
                    is_supporting = (title, sent_idx) in supporting_pairs
                    
                    if is_supporting:
                        question_positives.append(sentence)
                    else:
                        question_negatives.append(sentence)
            
            # Add all positive examples for this question
            for pos_sentence in question_positives:
                positive_questions.append(question)
                positive_contexts.append(pos_sentence)
            
            # Limit negative examples per question to avoid overwhelming dataset
            # Take at most 5 negative examples per question, or all if fewer
            max_negatives_per_question = 5
            selected_negatives = random.sample(
                question_negatives, 
                min(len(question_negatives), max_negatives_per_question)
            )
            
            for neg_sentence in selected_negatives:
                negative_questions.append(question)
                negative_contexts.append(neg_sentence)
        
        print(f"Before balancing: {len(positive_questions)} positive, {len(negative_questions)} negative")
        
        # Balance the dataset
        min_examples = min(len(positive_questions), len(negative_questions))
        
        if len(positive_questions) < len(negative_questions):
            # Use all positive examples and sample negatives
            balanced_positive_questions = positive_questions
            balanced_positive_contexts = positive_contexts
            
            indices = random.sample(range(len(negative_questions)), len(positive_questions))
            balanced_negative_questions = [negative_questions[i] for i in indices]
            balanced_negative_contexts = [negative_contexts[i] for i in indices]
        else:
            # Use all negative examples and sample positives
            balanced_negative_questions = negative_questions
            balanced_negative_contexts = negative_contexts
            
            indices = random.sample(range(len(positive_questions)), len(negative_questions))
            balanced_positive_questions = [positive_questions[i] for i in indices]
            balanced_positive_contexts = [positive_contexts[i] for i in indices]
        
        # Combine balanced examples
        questions = balanced_positive_questions + balanced_negative_questions
        contexts = balanced_positive_contexts + balanced_negative_contexts
        labels = [1] * len(balanced_positive_questions) + [0] * len(balanced_negative_questions)
        
        # Shuffle the combined dataset
        combined = list(zip(questions, contexts, labels))
        random.shuffle(combined)
        questions, contexts, labels = zip(*combined)
        questions, contexts, labels = list(questions), list(contexts), list(labels)
        
        print(f"Created {len(questions)} balanced training examples")
        print(f"Positive examples: {sum(labels)}")
        print(f"Negative examples: {len(labels) - sum(labels)}")
        
        return questions, contexts, labels
