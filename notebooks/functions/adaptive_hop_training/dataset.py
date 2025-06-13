import json
from datasets import load_dataset, Dataset
from datetime import datetime
CURRENT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    

class HotpotDataset:
    def __init__(self, dataset_config, train_file_path = None, val_file_path = None):
        self.dataset_config = dataset_config
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path

    
        if train_file_path and val_file_path:
            self.train_data, self.val_data = self.load_existing_dataset(train_file_path, val_file_path)
        else:
            self.train_data, self.val_data = self.create_dataset()

    def create_dataset(self):
        """Create and save HotpotQA dataset
        Gets a dictionary with dataset configuration and saves the training and validation splits to JSONL files."""

        # Load dataset
        dataset = load_dataset(
            self.dataset_config['dataset_name'],
            self.dataset_config['dataset_config'],
            trust_remote_code=True
        )

        # Extract training and validation splits
        train_data = dataset['train'][:self.dataset_config['train_size']]
        val_data = dataset['train'][self.dataset_config['train_size']:self.dataset_config['train_size'] + self.dataset_config['val_size']]

        if self.dataset_config['save_dataset']:
            # Generate filenames
            train_filename = f"hotpot_train_{self.dataset_config['train_size']}samples_{CURRENT_TIMESTAMP}.jsonl"
            val_filename = f"hotpot_val_{self.dataset_config['val_size']}samples_{CURRENT_TIMESTAMP}.jsonl"

            # Save to JSONL files
        def save_to_jsonl(data, filename):
            with open(filename, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')

        save_to_jsonl(train_data, train_filename)
        save_to_jsonl(val_data, val_filename)

        print(f"Dataset created and saved:")
        print(f"- Training: {train_filename} ({len(train_data['question'])} samples)")
        print(f"- Validation: {val_filename} ({len(val_data['question'])} samples)")

        return train_data, val_data

    def load_existing_dataset(train_filename=None, val_filename=None):
        """Load dataset from existing JSONL files"""
    
        # Default filenames if not provided
        if not train_filename:
            train_filename = "hotpot_train_5000samples_2025-06-11_18-31-48.jsonl"
        if not val_filename:
            val_filename = "hotpot_val_1000samples_2025-06-11_18-31-48.jsonl"
        
        try:
            # Load using datasets library
            train_dataset = Dataset.from_json(train_filename)
            val_dataset = Dataset.from_json(val_filename)
            
            print(f"Successfully loaded datasets:")
            print(f"- Training: {len(train_dataset)} samples from {train_filename}")
            print(f"- Validation: {len(val_dataset)} samples from {val_filename}")
            
            return train_dataset, val_dataset
            
        except FileNotFoundError as e:
            print(f"Error: Dataset files not found")
            print(f"Expected files: {train_filename}, {val_filename}")
            print("Please run the dataset creation cell first or provide correct filenames")
            return None, None
        
    
        
    def get_train_data(self):
        """Get training data"""
        if self.train_data is None:
            print("Training data not loaded. Please create or load the dataset first.")
            return None
        return self.train_data
    def get_val_data(self):
        """Get validation data"""
        if self.val_data is None:
            print("Validation data not loaded. Please create or load the dataset first.")
            return None
        return self.val_data

