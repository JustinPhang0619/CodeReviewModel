# Automated Code Review and Bug Detection Tool

A code review model utilising fine-tuned models of CodeT5 and CodeBERT. It is able to predict an error for code snippets and provide code suggestions and improvements.

Full GUI support and provided in EXE format for user convenience.

Due to large file constraint, most files that are used, such as the dataset, will be linked here.

The EXE file: 
https://drive.google.com/file/d/1kJTKsEhK3gwOEnK3xu3v_9st-PR-jRBK/view?usp=sharing

The first dataset is linked here:
https://github.com/michiyasunaga/BIFI

The second dataset is linked here:
https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca

The CodeBERT model used:
https://github.com/microsoft/CodeBERT

The CodeT5 model used:
https://huggingface.co/alexjercan/codet5-base-buggy-code-repair

After obtaining both the dataset and the models, a procedure is necessary to introduce error into the second dataset by using the "introduce_error.py" file.

Afterwards, simply put both models and dataset into the desired folders and alter the file paths of the training code.

Do not change the parameters set in the optimizer, scheduler and hyperparameters to obtain the same result.

Please do not hesitate to contact me if there are any inquiries!
