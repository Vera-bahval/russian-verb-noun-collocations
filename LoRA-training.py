#!/usr/bin/env python3
"""
Multitask Collocation Model Training Script

This script trains a language model on three collocation-related tasks:
1. Classification (determining if a phrase is a collocation)
2. Paraphrasing (replacing collocations with synonymous verbs)
3. Gap filling (completing sentences with missing collocations)

Usage:
    python multitask_collocation_trainer.py --train_path data/train.csv --test_path data/test.csv
"""

import os
import sys
import argparse
import pandas as pd
from datasets import Dataset, concatenate_datasets
import torch
from datetime import datetime
import random
import json
import time
from tqdm import tqdm

# Deep learning imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from sklearn.metrics import precision_score, recall_score, f1_score


def setup_directories(base_dir):
    """Create necessary directories for the project."""
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    data_dir = os.path.join(base_dir, "data")
    
    for directory in [base_dir, checkpoints_dir, models_dir, results_dir, data_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory {directory} created or already exists")
    
    return checkpoints_dir, models_dir, results_dir, data_dir


def load_data(train_path, test_path):
    """Load training and testing data from CSV files."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data file not found: {test_path}")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print("Data successfully loaded from files")
    print(f"Training dataset size: {len(train_df)}")
    print(f"Test dataset size: {len(test_df)}")
    
    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)


def prepare_multitask_data(dataset):
    """Prepare data for all three tasks simultaneously."""
    results = []

    # Process all examples in the dataset
    for example in dataset:
        # Task 1: Collocation classification (for all examples)
        classification_prompt = f'''[ЗАДАЧА: КЛАССИФИКАЦИЯ]
Ниже дано сочетание глагола и существительного.
Определи, является ли оно коллокацией. Ответь только 1 (если сочетание - коллокация) или только 0 (если сочетание - не коллокация).
В твоем ответе должна быть ТОЛЬКО соответствующая цифра, больше ничего.
Сочетание: {example['phrase'].lower()}
Ответ: '''

        classification_completion = "1" if example.get('label', 0) == 1 else "0"

        results.append({
            "prompt": classification_prompt,
            "completion": classification_completion,
            "collocation": example['phrase'],
            "task_type": "classification"
        })

        # Task 2: Paraphrasing (only for collocations with synonyms)
        if example.get('label', 0) == 1 and example.get('synonym') != 'Nan':
            paraphrase_prompt = f'''[ЗАДАЧА: ПЕРЕФРАЗИРОВАНИЕ]
Ниже дано сочетание глагола и существительного. Замени все сочетание на один глагол с таким же значением.
В ответе напиши ТОЛЬКО глагол, больше ничего.
Сочетание: {example['phrase'].lower()}
Ответ:'''

            results.append({
                "prompt": paraphrase_prompt,
                "completion": example['synonym'],
                "collocation": example['phrase'].lower(),
                "task_type": "paraphrase"
            })

        # Task 3: Gap filling (only for collocations with sentences)
        if (example.get('label', 0) == 1 and 
            example.get('gap_sent') != 'Nan' and 
            example.get('fill_sent') != 'Nan'):
            
            gap_prompt = f"""[ЗАДАЧА: ЗАПОЛНЕНИЕ ПРОПУСКА]
Ниже дано предложение. На месте [MASK] пропущено устойчивое сочетание русских глагола и существительного.
Напиши это сочетание. Сочетание, которое ты подберешь, должно подходит в предложении по смыслу и грамматически. Между глаголом и существительным может быть предлог.
В ответе напиши ТОЛЬКО сочетание глагола и существительного (и предлог между ниими, если нужно), больше ничего.
Предложение: {example['gap_sent']}
Ответ:"""

            results.append({
                "prompt": gap_prompt,
                "completion": example['fill_sent'].lower(),
                "collocation": example['phrase'].lower(),
                "task_type": "gap_filling"
            })

    # Shuffle data for better training
    random.shuffle(results)
    return Dataset.from_list(results)


def get_target_modules(model_name):
    """Get appropriate target modules for LoRA based on model architecture."""
    if "Qwen" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "YandexGPT" in model_name:
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def train_multitask_lora(model_name, train_dataset, output_dir, models_dir,
                         epochs=3, learning_rate=2e-4, save_every=100, batch_size=4,
                         use_quantization=True):
    """Train a multitask model using LoRA fine-tuning."""
    
    print(f"Starting training of model {model_name} on multitask data...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with or without quantization
    if use_quantization:
        print("Using 4-bit quantization for memory efficiency")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare model for 4-bit training
        model = prepare_model_for_kbit_training(model)
    else:
        print("Loading model without quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

    # Get target modules based on model architecture
    target_modules = get_target_modules(model_name)

    # LoRA configuration
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    print("Trainable parameters info:")
    model.print_trainable_parameters()

    # Tokenization function
    def tokenize_function(examples):
        # Combine prompt and completion for instruction format
        texts = []
        for i in range(len(examples["prompt"])):
            full_text = examples["prompt"][i] + examples["completion"][i]
            texts.append(full_text)

        # Tokenize with instruction formatting
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Create labels for training (shift by 1)
        tokenized["labels"] = tokenized["input_ids"].clone()

        # Mask prompt labels (only train on completion prediction)
        for i, prompt in enumerate(examples["prompt"]):
            prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
            prompt_length = prompt_tokens.shape[1]
            tokenized["labels"][i, :prompt_length] = -100

        return tokenized

    # Tokenize data
    print("Tokenizing data...")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )

    # Prepare directory names for saving checkpoints and model
    model_name_short = model_name.split("/")[-1]
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{output_dir}/{model_name_short}_multitask_{time_str}"
    final_model_dir = f"{models_dir}/{model_name_short}_multitask_{time_str}"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4 if batch_size < 4 else 1,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=save_every,
        logging_steps=10,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        save_total_limit=3,
        remove_unused_columns=False,
        seed=42
    )

    # Create Trainer
    print("Creating Trainer and starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save final model
    print(f"Saving model to {final_model_dir}...")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Save configuration info
    config_info = {
        "model_name": model_name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "quantization": use_quantization,
        "train_dataset_size": len(train_dataset),
        "lora_rank": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "target_modules": target_modules
    }

    with open(f"{final_model_dir}/training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, ensure_ascii=False, indent=2)

    print(f"Training completed. Model saved to {final_model_dir}")
    return final_model_dir


def evaluate_multitask_model(model_path, test_dataset, results_dir):
    """Evaluate the trained multitask model on test data."""
    
    print(f"Evaluating model from {model_path} on test dataset...")
    results = {}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Check if quantization was used during training
    try:
        with open(f"{model_path}/training_config.json", "r", encoding="utf-8") as f:
            config_info = json.load(f)
        use_quantization = config_info.get("quantization", False)
    except FileNotFoundError:
        use_quantization = False

    model_loading_params = {
        "pretrained_model_name_or_path": model_path,
        "device_map": "auto",
        "trust_remote_code": True
    }

    if use_quantization:
        print("Loading model with 4-bit quantization for evaluation")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_loading_params["quantization_config"] = bnb_config
        model_loading_params["torch_dtype"] = torch.bfloat16
    else:
        print("Loading model without quantization for evaluation")
        model_loading_params["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(**model_loading_params)
    
    # Initialize pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=50,
        do_sample=False
    )

    # Separate data by task types
    classification_examples = test_dataset.filter(lambda x: x["task_type"] == "classification")
    paraphrase_examples = test_dataset.filter(lambda x: x["task_type"] == "paraphrase")
    gap_filling_examples = test_dataset.filter(lambda x: x["task_type"] == "gap_filling")
    
    # Evaluate classification task
    true_labels_cls = []
    pred_labels_cls = []
    
    classification_output = f"{results_dir}/classification_results.txt"
    with open(classification_output, 'w') as f:
        for example in tqdm(classification_examples, desc="Evaluating classification"):
            prompt = example["prompt"]
            ground_truth = example["completion"]
            collocation = example["collocation"]
            output = pipe(prompt)[0]["generated_text"]
            prediction = output[len(prompt):].strip()
            f.write(f"{collocation}@{ground_truth}@{prediction}\n")
            f.flush()
            
            true_labels_cls.append(int(ground_truth))
            if '1' in prediction:
                pred = 1
            elif '0' in prediction:
                pred = 0
            else:
                pred = 0
            pred_labels_cls.append(pred)
    
    # Calculate classification metrics
    precision = precision_score(true_labels_cls, pred_labels_cls)
    recall = recall_score(true_labels_cls, pred_labels_cls)
    f1 = f1_score(true_labels_cls, pred_labels_cls)
    
    results["classification_metrics"] = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    print(f"Classification - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Evaluate paraphrase task
    paraphrase_output = f"{results_dir}/paraphrase_results.txt"
    with open(paraphrase_output, 'w') as f:
        correct = 0
        total = 0
        for example in tqdm(paraphrase_examples, desc="Evaluating paraphrases"):
            prompt = example["prompt"]
            ground_truth = example["completion"]
            collocation = example["collocation"]
            output = pipe(prompt)[0]["generated_text"]
            prediction = output[len(prompt):].strip().lower()
            f.write(f"{collocation}@{ground_truth}@{prediction}\n")
            f.flush()
            
            if ground_truth.lower() == prediction:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        results["paraphrase_accuracy"] = accuracy
        print(f"Paraphrase accuracy: {accuracy:.4f}")

    # Evaluate gap filling task
    gap_filling_output = f"{results_dir}/gap_filling_results.txt"
    with open(gap_filling_output, 'w') as f:
        correct = 0
        total = 0
        for example in tqdm(gap_filling_examples, desc="Evaluating gap filling"):
            prompt = example["prompt"]
            ground_truth = example["completion"]
            collocation = example["collocation"]
            output = pipe(prompt)[0]["generated_text"]
            prediction = output[len(prompt):].strip().lower()
            f.write(f"{collocation}@{ground_truth}@{prediction}\n")
            f.flush()
            
            if ground_truth.lower() == prediction:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        results["gap_filling_accuracy"] = accuracy
        print(f"Gap filling accuracy: {accuracy:.4f}")

    # Save results
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_path.split("/")[-1]
    results_file = f"{results_dir}/{model_name}_multitask_results_{time_str}.json"
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate multitask collocation model")
    parser.add_argument("--train_path", type=str, required=True, 
                        help="Path to training CSV file")
    parser.add_argument("--test_path", type=str, required=True, 
                        help="Path to test CSV file")
    parser.add_argument("--base_dir", type=str, default="./collocation_project",
                        help="Base directory for the project")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name to use for training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate for training")
    parser.add_argument("--use_quantization", action="store_true", default=True,
                        help="Use 4-bit quantization for memory efficiency")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and only evaluate existing model")
    parser.add_argument("--model_path", type=str,
                        help="Path to pre-trained model for evaluation (when skipping training)")
    
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Setup directories
    checkpoints_dir, models_dir, results_dir, data_dir = setup_directories(args.base_dir)

    # Load data
    train_dataset, test_dataset = load_data(args.train_path, args.test_path)

    # Prepare multitask data
    print("Preparing data for multitask learning...")
    train_multitask = prepare_multitask_data(train_dataset)
    test_multitask = prepare_multitask_data(test_dataset)

    print(f"Training examples: {len(train_multitask)}")
    print(f"Test examples: {len(test_multitask)}")

    if not args.skip_training:
        # Train the model
        print(f"\n===== Starting training of model {args.model_name.split('/')[-1]} =====")
        
        model_path = train_multitask_lora(
            model_name=args.model_name,
            train_dataset=train_multitask,
            output_dir=checkpoints_dir,
            models_dir=models_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            use_quantization=args.use_quantization
        )
    else:
        if not args.model_path:
            print("Error: --model_path is required when --skip_training is used")
            sys.exit(1)
        model_path = args.model_path

    # Evaluate the model
    print(f"\n===== Evaluating model =====")
    metrics = evaluate_multitask_model(
        model_path=model_path,
        test_dataset=test_multitask,
        results_dir=results_dir
    )

    print("\n===== Training and evaluation completed =====")
    print(f"Results saved in: {results_dir}")


if __name__ == "__main__":
    main()
