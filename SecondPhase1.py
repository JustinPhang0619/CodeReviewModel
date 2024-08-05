import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from datasets import load_metric

# Create checkpoint directory if it doesn't exist
checkpoint_dir = './SecondPhaseBestModel/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Load the dataset
def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

train_data_1 = load_jsonl(r'C:\VSCode Projects\Capstone\local_python_code_instructions_18k_alpaca\train\error_dataset.jsonl')
train_data_2 = load_jsonl(r'C:\VSCode Projects\Capstone\local_python_code_instructions_18k_alpaca\train\dataset_real.jsonl')

# Extract input and target code snippets
def extract_code_pairs(data1, data2):
    inputs = [entry['output'] for entry in data1]
    targets = [entry['output'] for entry in data2]
    return inputs, targets

inputs, targets = extract_code_pairs(train_data_1, train_data_2)

# Split the data into training and validation sets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, targets, test_size=0.2, random_state=42)

# Create a custom dataset class
class CodeDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=512):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target_text = self.targets[idx]

        input_encodings = self.tokenizer(
            input_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )
        target_encodings = self.tokenizer(
            target_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt'
        )

        input_ids = input_encodings['input_ids'].squeeze()
        attention_mask = input_encodings['attention_mask'].squeeze()
        labels = target_encodings['input_ids'].squeeze()

        # Replace padding token id's of the labels by -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("alexjercan/codet5-base-buggy-code-repair")
model = AutoModelForSeq2SeqLM.from_pretrained("alexjercan/codet5-base-buggy-code-repair")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create datasets and data loaders
train_dataset = CodeDataset(train_inputs, train_targets, tokenizer)
val_dataset = CodeDataset(val_inputs, val_targets, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=4e-5, betas=(0.9, 0.999), eps=1e-8)
num_epochs = 1
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=total_steps)

# Initialize metrics
bleu_metric = load_metric("bleu")
rouge_metric = load_metric("rouge")
meteor_metric = load_metric("meteor")

# Training loop with early stopping
def train_model(model, train_loader, optimizer, scheduler, device, epochs=1, gradient_accumulation_steps=4, early_stopping_patience=3):
    model.train()
    scaler = GradScaler()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        for batch_num, batch in enumerate(train_loader, 1):
            print(f'Epoch: {epoch+1}, Batch: {batch_num}/{len(train_loader)}')
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if batch_num % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            print(f'Epoch: {epoch+1}, Batch: {batch_num}/{len(train_loader)}, Loss: {loss.item() * gradient_accumulation_steps}')

        # Save the final state of the model
        save_path = './SecondPhaseTrainLast/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f'Model and tokenizer saved to {save_path}')
        
        val_loss, bleu_score, rouge_score, meteor_score = evaluate_model(model, val_loader, device, tokenizer)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, BLEU Score: {bleu_score}, ROUGE Score: {rouge_score}, METEOR Score: {meteor_score}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

def evaluate_model(model, val_loader, device, tokenizer):
    model.eval()
    total_loss = 0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['labels'].to(device)
            labels.extend(batch['labels'].tolist())

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                loss = outputs.loss
                total_loss += loss.item()
                preds.extend(torch.argmax(outputs.logits, dim=-1).tolist())

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Filter out -100 values from labels before decoding
    filtered_labels = [[token for token in label if token != -100] for label in labels]
    decoded_labels = tokenizer.batch_decode(filtered_labels, skip_special_tokens=True)

    bleu_score = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_score = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_score = meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return total_loss / len(val_loader), bleu_score, rouge_score, meteor_score

# Train and evaluate the model
train_model(model, train_loader, optimizer, scheduler, device, epochs=num_epochs, early_stopping_patience=3)

val_loss, bleu_score, rouge_score, meteor_score = evaluate_model(model, val_loader, device, tokenizer)
print(f'Validation Loss: {val_loss}')
print(f'BLEU Score: {bleu_score}')
print(f'ROUGE Score: {rouge_score}')
print(f'METEOR Score: {meteor_score}')

# Save the final state of the model
save_path = './SecondPhaseTrainLast/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f'Model and tokenizer saved to {save_path}')

# Load and save the best model
best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
model.load_state_dict(torch.load(best_model_path))
model.save_pretrained(os.path.join(save_path, 'best_model'))
tokenizer.save_pretrained(os.path.join(save_path, 'best_model'))
print(f'Best model saved to {os.path.join(save_path, "best_model")}')
