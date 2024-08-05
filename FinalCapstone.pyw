import json
import os
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import filedialog, Text, Scrollbar
import sys

# Helper function to get the correct path
def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load error advice
with open(get_resource_path("error_advice.json"), "r") as f:
    error_advice = json.load(f)

# Define paths
output_dir = get_resource_path("CapstonePart1")
label_encoder_path = os.path.join(output_dir, "label_encoder_pickle.pkl")
refinement_model_path = get_resource_path("CapstonePart2")

# Load models and tokenizers
classification_model = AutoModelForSequenceClassification.from_pretrained(output_dir)
classification_tokenizer = AutoTokenizer.from_pretrained(output_dir)
with open(label_encoder_path, "rb") as file:
    label_encoder = pickle.load(file)
refinement_tokenizer = AutoTokenizer.from_pretrained(refinement_model_path)
refinement_model = T5ForConditionalGeneration.from_pretrained(refinement_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classification_model.to(device)
refinement_model.to(device)

def predict_code_error(code_snippet, model, tokenizer, label_encoder, max_length=250):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            code_snippet,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).cpu().numpy()

        decoded_prediction = label_encoder.inverse_transform(prediction)
        probabilities = probabilities.cpu().numpy().flatten()
        class_probabilities = {label: prob for label, prob in zip(label_encoder.classes_, probabilities)}

    return decoded_prediction[0], class_probabilities

def refine_code(input_code, model, tokenizer, max_length=300):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(input_code, return_tensors="pt").to(device)
        output_ids = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
        refined_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return refined_code

def get_file_path():
    file_path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
    if file_path:
        with open(file_path, 'r') as file:
            code_snippet = file.read()
            code_input.delete(1.0, tk.END)
            code_input.insert(tk.END, code_snippet)
        remove_placeholder(event=None)

def analyze_code():
    code_snippet = code_input.get("1.0", tk.END).strip()
    if code_snippet == "" or code_snippet == "Please add your desired code snippet or open file to import code here":
        result_output.config(state=tk.NORMAL)
        result_output.delete(1.0, tk.END)
        result_output.insert(tk.END, "No code input provided.\n")
        result_output.config(state=tk.DISABLED)
        return

    result_output.config(state=tk.NORMAL)
    result_output.delete(1.0, tk.END)
    result_output.config(fg="black")
    
    predicted_error_message, class_probabilities = predict_code_error(code_snippet, classification_model, classification_tokenizer, label_encoder)
    result_output.insert(tk.END, f"Predicted Error Message: {predicted_error_message}\n")

    if predicted_error_message in error_advice:
        result_output.insert(tk.END, f"\n{error_advice[predicted_error_message]}\n")

    # result_output.insert(tk.END, "\nClass Probabilities:\n")
    # for class_label, probability in class_probabilities.items():
    #     result_output.insert(tk.END, f"{class_label}: {probability:.4f}\n")
    result_output.insert(tk.END, "\n---------------------------------------------\n")
    refined_code = refine_code(code_snippet, refinement_model, refinement_tokenizer)
    result_output.insert(tk.END, "\nOriginal Code:\n")
    result_output.insert(tk.END, code_snippet + "\n")

    result_output.insert(tk.END, "\nPotential Solution:\n")
    result_output.insert(tk.END, refined_code + "\n")

    result_output.config(state=tk.DISABLED)

def add_placeholder(event=None):
    if code_input.get("1.0", tk.END).strip() == "":
        code_input.insert("1.0", "Please add your desired code snippet or open file to import code here.")
        code_input.config(fg="grey")

def remove_placeholder(event=None):
    if code_input.get("1.0", tk.END).strip() == "Please add your desired code snippet or open file to import code here.":
        code_input.delete("1.0", tk.END)
        code_input.config(fg="black")

def add_result_placeholder(event=None):
    if result_output.get("1.0", tk.END).strip() == "":
        result_output.config(state=tk.NORMAL)
        result_output.insert("1.0", "The analysis will be shown here.")
        result_output.config(fg="grey", state=tk.DISABLED)

def remove_result_placeholder(event=None):
    if result_output.get("1.0", tk.END).strip() == "The analysis will be shown here.":
        result_output.config(state=tk.NORMAL)
        result_output.delete("1.0", tk.END)
        result_output.config(fg="black", state=tk.DISABLED)

# Set up the GUI
root = tk.Tk()
root.title("Automated Code Review and Bug Detection Tool")
root.configure(bg='#ADD8E6')

frame = tk.Frame(root, bg='#ADD8E6')
frame.pack(pady=20)

scrollbar = Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

code_input = Text(frame, height=20, width=80, padx=10, pady=10, wrap=tk.WORD, yscrollcommand=scrollbar.set, bg='#E0FFFF', fg='black')  # Light blue background
code_input.pack()
code_input.insert("1.0", "Please add your desired code snippet or open file to import code here.")
code_input.config(fg="grey")

code_input.bind("<FocusIn>", remove_placeholder)
code_input.bind("<FocusOut>", add_placeholder)

scrollbar.config(command=code_input.yview)

btn_frame = tk.Frame(root, bg='#ADD8E6')
btn_frame.pack(pady=10)

file_btn = tk.Button(btn_frame, text="Open File", command=get_file_path, bg='#B0C4DE', fg='black')  # Light steel blue background
file_btn.grid(row=0, column=0, padx=10)

analyze_btn = tk.Button(btn_frame, text="Analyze Code", command=analyze_code, bg='#B0C4DE', fg='black')  # Light steel blue background
analyze_btn.grid(row=0, column=1, padx=10)

result_frame = tk.Frame(root, bg='#ADD8E6')
result_frame.pack(pady=10)

scrollbar2 = Scrollbar(result_frame)
scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)

result_output = Text(result_frame, height=20, width=80, padx=10, pady=10, wrap=tk.WORD, yscrollcommand=scrollbar2.set, bg='#E0FFFF', fg='black')  # Light blue background
result_output.pack()
result_output.insert("1.0", "The analysis will be shown here.")
result_output.config(fg="grey", state=tk.DISABLED)

result_output.bind("<FocusIn>", remove_result_placeholder)
result_output.bind("<FocusOut>", add_result_placeholder)

scrollbar2.config(command=result_output.yview)

root.mainloop()
