from datasets import load_dataset, concatenate_datasets
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import torch

# --- 1. Load Datasets ---
print("Loading Hugging Face dataset...")
hf_dataset = load_dataset("Programmer-RD-AI/genz-slang-pairs-1k")

print("Loading local CSV dataset...")
local_dataset = load_dataset("csv", data_files="genz_normal_sentences.csv")

# --- 1a. Combine Datasets ---
# We assume both datasets have 'normal' and 'gen_z' columns
# We take the 'train' split from both and combine them
print("Combining datasets...")
combined_dataset = concatenate_datasets(
    [hf_dataset["train"], local_dataset["train"]]
)
print(f"Total combined samples: {len(combined_dataset)}")

# check what columns exist (should be 'normal', 'gen_z')
print(f"Columns: {combined_dataset.column_names}")

# --- 1b. Split 70/30 ---
split_dataset = combined_dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# --- 2. Pick Model ---
model_name = "t5-small"  # can change to t5-base if you have more VRAM
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# --- 3. Preprocessing Function ---
# This function is perfect, it already creates both sets of fields
def preprocess(example):
    # format inputs as T5 expects
    example["eng_to_genz_input"] = "translate English to GenZ: " + example["normal"]
    example["genz_to_eng_input"] = "translate GenZ to English: " + example["gen_z"]
    example["eng_to_genz_target"] = example["gen_z"]
    example["genz_to_eng_target"] = example["normal"]
    return example

train_dataset = train_dataset.map(preprocess)
eval_dataset = eval_dataset.map(preprocess)

# --- 4. Tokenization (NOW FOR BOTH DIRECTIONS) ---
print("Tokenizing for English to GenZ...")
def tokenize_eng_to_genz(examples):
    model_inputs = tokenizer(
        examples["eng_to_genz_input"], max_length=128, truncation=True
    )
    labels = tokenizer(
        examples["eng_to_genz_target"], max_length=128, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_e2g = train_dataset.map(tokenize_eng_to_genz, batched=True)
tokenized_eval_e2g = eval_dataset.map(tokenize_eng_to_genz, batched=True)

print("Tokenizing for GenZ to English...")
def tokenize_genz_to_eng(examples):
    model_inputs = tokenizer(
        examples["genz_to_eng_input"], max_length=128, truncation=True
    )
    labels = tokenizer(
        examples["genz_to_eng_target"], max_length=128, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_g2e = train_dataset.map(tokenize_genz_to_eng, batched=True)
tokenized_eval_g2e = eval_dataset.map(tokenize_genz_to_eng, batched=True)

print("Combining tokenized datasets...")
# Combine both sets of training and eval data
final_train_dataset = concatenate_datasets([tokenized_train_e2g, tokenized_train_g2e])
final_eval_dataset = concatenate_datasets([tokenized_eval_e2g, tokenized_eval_g2e])

# Shuffle the combined datasets to mix the tasks
final_train_dataset = final_train_dataset.shuffle(seed=42)
final_eval_dataset = final_eval_dataset.shuffle(seed=42)

print(f"Total training examples (both directions): {len(final_train_dataset)}")
print(f"Total evaluation examples (both directions): {len(final_eval_dataset)}")


# --- 5. Training Arguments ---
training_args = TrainingArguments(
    output_dir="./models/t5-translator", # <-- Updated output directory
    eval_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
)

# --- 6. Trainer Setup ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_train_dataset, # <-- Use the final combined dataset
    eval_dataset=final_eval_dataset,   # <-- Use the final combined dataset
    tokenizer=tokenizer,  # <-- Corrected this argument
    data_collator=data_collator,
)

# --- 7. Train! ---
trainer.train()

# --- 8. Save the Fine-tuned Model ---
model.save_pretrained("./models/t5-translator") # <-- Updated save directory
tokenizer.save_pretrained("./models/t5-translator") # <-- Updated save directory

print("✅ Fine-tuning complete! Model saved to ./models/t5-translator")
