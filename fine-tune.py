from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import torch

# --- 1. Load Dataset ---
dataset = load_dataset("Programmer-RD-AI/genz-slang-pairs-1k")

# check what columns exist
print(dataset["train"].column_names)

# --- 1a. Split 70/30 ---
split_dataset = dataset["train"].train_test_split(test_size=0.3, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"Training samples: {len(train_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# --- 2. Pick Model ---
model_name = "t5-small"  # can change to t5-base if you have more VRAM
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# --- 3. Preprocessing Function ---
def preprocess(example):
    # format inputs as T5 expects
    example["eng_to_genz_input"] = "translate English to GenZ: " + example["normal"]
    example["genz_to_eng_input"] = "translate GenZ to English: " + example["gen_z"]
    example["eng_to_genz_target"] = example["gen_z"]
    example["genz_to_eng_target"] = example["normal"]
    return example

train_dataset = train_dataset.map(preprocess)
eval_dataset = eval_dataset.map(preprocess)

# --- 4. Tokenization ---
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["eng_to_genz_input"], max_length=128, truncation=True
    )
    labels = tokenizer(
        examples["eng_to_genz_target"], max_length=128, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# --- 5. Training Arguments ---
training_args = TrainingArguments(
    output_dir="./models/eng_to_genz",
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
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# --- 7. Train! ---
trainer.train()

# --- 8. Save the Fine-tuned Model ---
model.save_pretrained("./models/eng_to_genz")
tokenizer.save_pretrained("./models/eng_to_genz")

print("✅ Fine-tuning complete! Model saved to ./models/eng_to_genz")

