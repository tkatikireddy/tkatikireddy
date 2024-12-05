# Web-Mining-Final-Project

This project focuses on preprocessing code datasets, fine-tuning the Salesforce CodeT5 model, and evaluating its performance on the task of code summarization. The project showcases the process of transforming raw datasets into actionable insights through the use of Hugging Face Transformers, Pandas, and other libraries.

## Table of Contents

1. [Introduction](#introduction)
2. [Libraries Used](#libraries-used)
3. [Setup](#setup)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Evaluation](#evaluation)
8. [Generated Results](#generated-results)
9. [Usage](#usage)
10. [Conclusion](#conclusion)

## Introduction

This project demonstrates the end-to-end process of data preprocessing, fine-tuning, and evaluating the CodeT5 model for code summarization. The dataset contains code snippets paired with their natural language summaries. The fine-tuned model generates semantic code summaries, aiding in better code understanding.

## Libraries Used

- **transformers**: For pre-trained CodeT5 models and fine-tuning.
- **datasets**: For dataset handling and transformations.
- **pandas**: For data manipulation.
- **torch**: Core deep learning library for model training.
- **os** and **zipfile**: Utility libraries for file handling.
- **requests**: For downloading additional resources.

## Setup

1. Install dependencies:
   ```bash
   pip install transformers datasets pandas

2. Place the dataset file (dataset.zip) in the appropriate directory.

## Data Preprocessing

### Steps:

1. Extract and preprocess the dataset to convert JSONL files into structured CSV files.
2. Tokenize code snippets and docstrings for training the model.

### Scripts:

1. WebMining_Final_Project.py: Main script for dataset preprocessing and model training.

## Model Architecture

### Fine-Tuned Model

1. Pre-trained Salesforce/codet5-base fine-tuned on custom datasets.
2. Enhanced for the task of semantic code summarization.

## Training the Model

Hugging Face's Trainer API simplifies fine-tuning with components such as:
- Loss function: Cross-entropy loss.
- Optimizer: AdamW.

Training details are included in the main script: WebMining_Final_Project.py.

## Evaluation

The fine-tuned model is evaluated on the validation dataset. Key metrics include:
- Accuracy: Semantic alignment of summaries.
- BLEU Score: Measure of the similarity between generated and reference summaries.

## Generated Results

Below is an example of the model's output:

### Input Code:

```python
def add(a, b):
    return a + b
```
### Generated Summary:

```text
Function to add two numbers and return the result.
```
## Usage

### Preprocess the Datasets

Run the preprocessing steps in the main script:
```bash
python WebMining_Final_Project.py
```
### Train the Model

Fine-tuning is included within the main script.

### Use the Model for Inference

An example for generating summaries:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./final_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

code = "def add(a, b):\n    return a + b"
input_text = "summarize: " + code
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
description = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Description:", description)
```
## Conclusion

The fine-tuned CodeT5 model:
- Demonstrates improved semantic understanding for code summarization.
- Can be extended to support additional programming languages and tasks.

## Future Work

1. Expanding the dataset to include more languages and diverse code snippets.
2. Optimizing hyperparameters for better model performance.
3. Exploring alternative transformer models for code understanding tasks.

## License

This project is licensed under the MIT License.
```css
This `README.md` template captures the details of your project with sections similar to the example provided. Adjust the content to match your specific project details and workflows.
```
