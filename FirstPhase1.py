import pickle
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np
import json
import os
import logging
from torch.utils.tensorboard import SummaryWriter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and preprocess data
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {e}")
        raise

bad_data = load_data(r"Datasets\data_minimal\data\orig_bad_code\orig.bad.json")
good_data = load_data(r"Datasets\data_minimal\data\orig_good_code\orig.goood.json")

bad_code_snippets = [bad_data[id]["code_string"] for id in bad_data]
bad_error_messages = [bad_data[id]["err_obj"]["msg"] for id in bad_data]

good_code_snippets = [good_data[id]["code_string"] for id in good_data]
good_error_messages = ["The code looks good!"] * len(good_code_snippets)

all_code_snippets = bad_code_snippets + good_code_snippets
all_error_messages = bad_error_messages + good_error_messages

# Label encoding
label_encoder = LabelEncoder()
all_error_labels = label_encoder.fit_transform(all_error_messages)

# Convert to Hugging Face Dataset
data = {
    'text': all_code_snippets,
    'labels': all_error_labels
}
dataset = Dataset.from_dict(data)

# Split dataset
train_dataset, eval_dataset = dataset.train_test_split(test_size=0.2).values()

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=len(label_encoder.classes_))

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Convert datasets to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create a weighted sampler
class_counts = np.bincount(train_dataset['labels'])
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_dataset['labels']]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=sampler, batch_size=8)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8)

# Define training arguments with gradient clipping and logging
output_dir = './results'
learning_rate = 2e-5
num_train_epochs = 3
weight_decay = 0.01
max_grad_norm = 1.0
logging_dir = './logs'
logging_steps = 10
save_total_limit = 2

# Initialize optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
num_training_steps = num_train_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Setup TensorBoard logging
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
writer = SummaryWriter(log_dir=logging_dir)

# Train model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

global_step = 0
best_eval_accuracy = 0
best_model_state_dict = None

for epoch in range(num_train_epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        print(f"Epoch {epoch+1} - Step {step+1}/{len(train_loader)}", end='\r')
        batch = {k: v.to(device) for k, v in batch.items()}
        try:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if global_step % logging_steps == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1
        except Exception as e:
            logger.error(f"Error during training step: {e}")
            continue

    # Create dir if not exists
    if not os.path.exists('./FirstPhaseTrainLast'):
        os.makedirs('./FirstPhaseTrainLast')
    model.save_pretrained('./FirstPhaseTrainLast')
    tokenizer.save_pretrained('./FirstPhaseTrainLast')
    with open('./FirstPhaseTrainLast/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Evaluate model
    model.eval()
    eval_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        try:
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                eval_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += predictions.size(0)
        except Exception as e:
            logger.error(f"Error during evaluation step: {e}")
            continue
    
    eval_loss /= len(eval_loader)
    eval_accuracy = correct_predictions / total_predictions
    writer.add_scalar("eval/loss", eval_loss, epoch)
    writer.add_scalar("eval/accuracy", eval_accuracy, epoch)
    print(f"Epoch {epoch+1} - Eval Loss: {eval_loss:.4f} - Eval Accuracy: {eval_accuracy:.4f}")

    # Save best model
    if eval_accuracy > best_eval_accuracy:
        best_eval_accuracy = eval_accuracy
        best_model_state_dict = model.state_dict()
        if not os.path.exists('./FirstPhaseBestModel'):
            os.makedirs('./FirstPhaseBestModel')
        model.save_pretrained('./FirstPhaseBestModel')
        tokenizer.save_pretrained('./FirstPhaseBestModel')
        with open('./FirstPhaseBestModel/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)

# Save final model
if best_model_state_dict is not None:
    model.load_state_dict(best_model_state_dict)

# Save training arguments
training_args = {
    'output_dir': output_dir,
    'learning_rate': learning_rate,
    'num_train_epochs': num_train_epochs,
    'weight_decay': weight_decay,
    'max_grad_norm': max_grad_norm,
    'logging_dir': logging_dir,
    'logging_steps': logging_steps,
    'save_total_limit': save_total_limit
}
with open('./FirstPhaseTrainLast/training_args.json', 'w') as f:
    json.dump(training_args, f)

# Save config and results to JSON for reproducibility
config = {
    'model_name': 'microsoft/codebert-base',
    'tokenizer_name': 'microsoft/codebert-base',
    'num_labels': len(label_encoder.classes_),
    'training_args': training_args,
    'results': {
        'eval_loss': eval_loss,
        'eval_accuracy': best_eval_accuracy
    }
}
with open('./FirstPhaseTrainLast/config.json', 'w') as f:
    json.dump(config, f)

print(f"Training complete. Best model saved to ./FirstPhaseBestModel and final model to ./FirstPhaseTrainLast.")
