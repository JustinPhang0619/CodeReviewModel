import os
import json
import pickle
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Define paths to datasets
error_dataset_path = 'C:/VSCode Projects/Capstone/local_python_code_instructions_18k_alpaca/train/error_dataset_with_labels.jsonl'
real_dataset_path = 'C:/VSCode Projects/Capstone/local_python_code_instructions_18k_alpaca/train/dataset_real.jsonl'

# Define model directory
model_dir = './CapstonePart1'

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained(model_dir)
model = RobertaForSequenceClassification.from_pretrained(model_dir)

# Load label encoder
with open(os.path.join(model_dir, 'label_encoder_pickle.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

# Define a dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['output']
        label = item.get('errors', ['The code looks good!'])[0]
        encoded_input = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoded_input['input_ids'].squeeze()
        attention_mask = encoded_input['attention_mask'].squeeze()
        label = self.label_encoder.transform([label])[0]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

# Function to preprocess and tokenize the data
def preprocess_data(dataset_path, tokenizer, label_encoder):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    processed_data = []
    for item in data:
        input_text = item['output']
        label = item.get('errors', ['The code looks good!'])[0]
        processed_data.append({'output': input_text, 'errors': [label]})
    return processed_data

# Preprocess both datasets
error_dataset = preprocess_data(error_dataset_path, tokenizer, label_encoder)
real_dataset = preprocess_data(real_dataset_path, tokenizer, label_encoder)

# Combine datasets
combined_dataset = error_dataset + real_dataset

# Split data into train and validation sets
train_data, val_data = train_test_split(combined_dataset, test_size=0.1, random_state=42)

# Create DataLoader
train_dataset = CustomDataset(train_data, tokenizer, label_encoder)
val_dataset = CustomDataset(val_data, tokenizer, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Extract labels from training data for class weights computation
all_labels = [item['label'].item() for item in train_dataset]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Compute class weights
unique_labels = np.unique(all_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Fine-tuning parameters
epochs = 2
lr = 2e-5
warmup_steps = 0.1 * len(train_loader)  # 10% of total steps

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_loader) * epochs)

# Fine-tune the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    print(f"Epoch {epoch + 1}/{epochs}")

    # Training loop with progress bar
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Apply class weights to the loss
        weighted_loss = loss * class_weights[labels].mean()
        total_train_loss += weighted_loss.item()

        weighted_loss.backward()
        optimizer.step()
        scheduler.step()

    average_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Train Loss: {average_train_loss:.4f}")

    if not os.path.exists('./CapstonePart1TRAIN'):
        os.makedirs('./CapstonePart1TRAIN')
    model.save_pretrained('./CapstonePart1TRAIN')
    tokenizer.save_pretrained('./CapstonePart1TRAIN')
    with open('./CapstonePart1TRAIN/label_encoder_pickle.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Validation
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Apply class weights to the loss
            weighted_loss = loss * class_weights[labels].mean()
            total_val_loss += weighted_loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    average_val_loss = total_val_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Epoch {epoch + 1}/{epochs}, Average Validation Loss: {average_val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    if not os.path.exists('./CapstonePart1VAL'):
        os.makedirs('./CapstonePart1VAL')
    model.save_pretrained('./CapstonePart1VAL')
    tokenizer.save_pretrained('./CapstonePart1VAL')
    with open('./CapstonePart1VAL/label_encoder_pickle.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

# Save the fine-tuned model
if not os.path.exists('./CapstonePart1Done'):
    os.makedirs('./CapstonePart1Done')
model.save_pretrained('./CapstonePart1Done')
tokenizer.save_pretrained('./CapstonePart1Done')
with open('./CapstonePart1Done/label_encoder_pickle.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
