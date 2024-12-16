from dataclasses import asdict

import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config import DataConfig, ModelConfig, TrainConfig
from src.datamodules import GPTChessDataModule, GPTChessDataset
# from src.llama import LlamaChessLightning
from transformers.models.llama.configuration_llama import LlamaConfig
import os
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,BitsAndBytesConfig, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import bitsandbytes as bnb
from functools import partial

data_config = DataConfig(
        dataset_path="data/stockfish",
        file_path="stockfish_dataset_blocks.zip",
        batch_size=2,
        num_workers=24,
    )

train_config = TrainConfig(
    max_epochs=50,
    val_check_interval=0.01,
    log_every_n_steps=10,
    overfit_batches=0,
    checkpoint_path="checkpoints/",
    checkpoint_interval=10000,
    wandb_project="chessgpt",
    wandb_tags=["runpod_stockfish_run"],
    gradient_accumulation_steps=2,
)

def load_model(model_name):

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def train(model, tokenizer, dataset, output_dir):
    # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
    model.gradient_checkpointing_enable()

    # 2 - Using the prepare_model_for_kbit_training method from PEFT
    model = prepare_model_for_kbit_training(model)

    # Create PEFT config for these modules and wrap the model to PEFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"], 
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False 
    
    # Print information models
    cuda_available = torch.cuda.is_available()    
    print("CUDA (GPU) available:", cuda_available)
    print_trainable_parameters(model)
    
    # Training parameters
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            num_train_epochs=50,
            warmup_steps=10,
            adam_beta1=0.9,
            adam_beta2=0.95,    
            weight_decay=0.0,
            learning_rate=6e-4,
            max_steps=600000,
            lr_scheduler_type="cosine",
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Launch training
    print("Training started!")
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)    
    
    # Saving model
    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    
    # Free memory for merging weights
    del model
    del trainer
    torch.cuda.empty_cache()

# def create_prompt_formats(sample):
#     """
#     Format various fields of the sample ('instruction', 'context', 'response')
#     Then concatenate them using two newline characters 
#     :param sample: Sample dictionnary
#     """

#     INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
#     INSTRUCTION_KEY = "### Instruction:"
#     INPUT_KEY = "Input:"
#     RESPONSE_KEY = "### Response:"
#     END_KEY = "### End"
    
#     blurb = f"{INTRO_BLURB}"
#     instruction = f"{INSTRUCTION_KEY}\n{sample['instruction']}"
#     input_context = f"{INPUT_KEY}\n{sample['context']}" if sample["context"] else None
#     response = f"{RESPONSE_KEY}\n{sample['response']}"
#     end = f"{END_KEY}"
    
#     parts = [part for part in [blurb, instruction, input_context, response, end] if part]

#     formatted_prompt = "\n\n".join(parts)
    
#     sample["text"] = formatted_prompt

#     return sample
def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["transcript"],
        max_length=max_length,
        truncation=True,
    )


# SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset: str):
    """Format & tokenize it so it is ready for training
    :param tokenizer (AutoTokenizer): Model Tokenizer
    :param max_length (int): Maximum number of tokens to emit from tokenizer
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    print(dataset)
    # dataset = dataset.map(create_prompt_formats)
    
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["transcript"],
    )

    dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)
    return dataset
    

model_name = "meta-llama/Llama-3.2-1B"
model, tokenizer = load_model(model_name)

dataset = load_dataset("data/stockfish", data_files="stockfish_dataset_blocks.zip", split="train")
dataset = preprocess_dataset(tokenizer, get_max_length(model), dataset.select(range(1000)))
output_dir = "checkpoints/llama-3.2-chessgpt-4bit"
train(model, tokenizer, dataset, output_dir)    