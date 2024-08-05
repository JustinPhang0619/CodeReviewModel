import json
import random
import os

# Function to introduce errors into the code
def introduce_errors(code, num_errors=1):
    lines = code.split('\n')
    
    errors = {
        'expected an indented block': [
            lambda code: code + "\n    # expected an indented block",
            lambda code: code + "\n:    pass  # expected an indented block",
            lambda code: code + "\n    raise NotImplementedError",
        ],
        'invalid syntax': [
            lambda code: code.replace('def ', 'deff ', 1),
            lambda code: code.replace('return', 'retrun', 1),
            lambda code: code.replace('if ', 'iff ', 1),
            # Mismatched or missing quotes
            lambda code: code.replace('"', '', 1),
            lambda code: code.replace("'", '"', 1),
            # Incorrect operator usage
            lambda code: code.replace('==', '=', 1),
            lambda code: code.replace('+=', '+', 1),
            # Improper function call
            lambda code: code.replace('(', '', 1),
            lambda code: code.replace(')', '', 1),
            lambda code: code.replace(', ', '', 1),
            # Incorrect keyword usage
            lambda code: "return\n" + code,
            lambda code: "break\n" + code,
            # Unfinished statements
            lambda code: code + "\n    a = ",
            lambda code: code + "\n    if x",
            # Improper import statements
            lambda code: code.replace('import', 'imprt', 1),
            lambda code: code.replace('import ', 'importing ', 1),
            # Syntax errors in data structures
            lambda code: code.replace(',', '', 1),
            lambda code: code.replace('[', '(', 1),
            lambda code: code.replace(']', ')', 1),
        ],
        'unbalanced (){}[]': [
            lambda code: code + ")",
            lambda code: code.replace('(', '', 1),
            lambda code: code.replace(')', '', 1),
        ],
        'unexpected indent': [
            lambda code: "    " + code,
            lambda code: "        " + code,
            lambda code: " " + code,
        ],
        'unexpected unindent': [
            lambda code: code.replace('\n', '\n    ', 1),
            lambda code: code.replace('    ', '', 1),
            lambda code: code.replace('        ', '', 1),
        ],
        'unindent does not match any outer indentation level': [
            lambda code: code + "\n    else:",
            lambda code: code + "\n    except:",
            lambda code: code + "\nelse:",
        ],
    }
    
    error_types = []
    for _ in range(num_errors):
        error_type = random.choice(list(errors.keys()))
        error_variation = random.choice(errors[error_type])
        lines = error_variation('\n'.join(lines)).split('\n')
        error_types.append(error_type)
    
    return '\n'.join(lines), error_types

# Ensure the output directory exists
output_dir = os.path.dirname(r'C:\VSCode Projects\Capstone\local_python_code_instructions_18k_alpaca')
os.makedirs(output_dir, exist_ok=True)

# Read the JSONL file and introduce errors
input_file_path = r'C:\VSCode Projects\Capstone\local_python_code_instructions_18k_alpaca\train\dataset_real.jsonl'
output_file_path = r'C:\VSCode Projects\Capstone\local_python_code_instructions_18k_alpaca\train\error_dataset_with_labels.jsonl'

try:
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        for line in infile:
            try:
                json_obj = json.loads(line)
                
                # Introduce errors into the 'output' field if it exists
                if 'output' in json_obj and json_obj['output'].strip():
                    modified_code, error_labels = introduce_errors(json_obj['output'], 1)
                    json_obj['output'] = modified_code
                    json_obj['errors'] = error_labels
                
                # Write the modified JSON object to the output file
                outfile.write(json.dumps(json_obj) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

print(f"Modified JSONL file saved to {output_file_path}")
