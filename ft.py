import os
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, MistralForCausalLM
import transformers
import matplotlib.pyplot as plt
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import wandb
from datetime import datetime


fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

## set up wandb
wandb.login()

wandb_project = "fyp"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project


## Load the data
df = pd.read_pickle('data/ghost_stories/fine_tuning_prompts/top10books_500prompts.pkl')

## Split the data
usable_df = df.sample(frac=0.2, random_state=42)
non_usable_df = df.drop(usable_df.index)
non_usable_df.to_pickle('data/ghost_stories/fine_tuning_prompts/top10books_500prompts_nonusable.pkl')

df = usable_df
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

## Print the lengths
print(f"train df length: {len(train_df)}")
print(f"text df length: {len(val_df)}")


## Load the model
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = MistralForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, attn_implementation="flash_attention_2")

## Tokenize the data
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=2560,
    padding_side="left",
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(ex):
    result = tokenizer(ex, truncation=True, max_length=2560, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

def create_and_tokenize_prompt(row):
    prompt = row['generated_text'].strip()
    answer = row['paragraph'].strip()
    full_prompt = f"""[INST] You are a creative writing teacher specialising in the genre of ghost stories. {prompt} [/INST]
    {answer}"""
    return tokenize_function(full_prompt)

tokenized_train_df = train_df.apply(create_and_tokenize_prompt, axis=1)
tokenized_val_df = val_df.apply(create_and_tokenize_prompt, axis=1)

# strip the generated text and paragraph
train_df['generated_text'] = train_df['generated_text'].str.strip()


print(tokenized_train_df.head())
print(tokenized_train_df[0]['input_ids'])
print(tokenized_train_df[0]['labels'])
print(tokenized_train_df[0]['attention_mask'])

## set up LoRA

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

# Apply the accelerator. You can comment this out to remove the accelerator.
model = accelerator.prepare_model(model)

project = "fyp"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

if torch.cuda.device_count() > 1: # If more than 1 GPU
    print("Using DataParallel")
    model.is_parallelizable = True
    model.model_parallel = True

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_df,
    eval_dataset=tokenized_val_df,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=100,
        learning_rate=2.5e-5, # Want about 10x smaller than the Mistral learning rate
        logging_steps=25,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=25,                # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=25,               # Evaluate and save checkpoints every 50 steps
        do_eval=True,                # Perform evaluation at the end of training
        report_to="wandb",           # Comment this out if you don't want to use weights & baises
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"          # Name of the W&B run (optional)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()





