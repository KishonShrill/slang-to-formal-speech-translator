from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import torch
import pandas as pd

# --- 1. Load Dataset ---
dataset = load_dataset("Programmer-RD-AI/genz-slang-pairs-1k")
csv_path = "data/gen_zz_words.csv"

df_slang = pd.read_csv(csv_path)

df_slang_hf = df_slang[['Word/Phrase', 'Definition']].rename(columns={'Definition': 'normal', 'Word/Phrase': 'gen_z'})

# Wrap definitions in a template to make them phrase-like
df_slang_hf['normal'] = "The meaning of this slang is: " + df_slang_hf['normal']

new_dataset = Dataset.from_pandas(df_slang_hf)

# check what columns exist
print(dataset["train"].column_names)

# --- 1a. Split 70/30 ---
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

augmented_dataset = concatenate_datasets([train_dataset, new_dataset]).shuffle(seed=42)



print(f"Total training samples after augmentation: {len(augmented_dataset)}")
print(f"Evaluation samples: {len(eval_dataset)}")

# --- 2. Pick Model ---
model_name = "t5-small" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# --- 3. Preprocessing Function ---
def preprocess(example):
    example["eng_to_genz_input"] = "translate English to GenZ: " + example["normal"]
    example["genz_to_eng_input"] = "translate GenZ to English: " + example["gen_z"]
    example["eng_to_genz_target"] = example["gen_z"]
    example["genz_to_eng_target"] = example["normal"]
    return example

augmented_dataset = augmented_dataset.map(preprocess)
eval_dataset = eval_dataset.map(preprocess)


# --- 4. Tokenization ---
# English → GenZ
def tokenize_eng_to_genz(examples):
    inputs = tokenizer(examples["eng_to_genz_input"], max_length=128, truncation=True)
    labels = tokenizer(examples["eng_to_genz_target"], max_length=128, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_eng_to_genz = augmented_dataset.map(tokenize_eng_to_genz, batched=True)
eval_eng_to_genz = eval_dataset.map(tokenize_eng_to_genz, batched=True)

# GenZ → English
def tokenize_genz_to_eng(examples):
    inputs = tokenizer(examples["genz_to_eng_input"], max_length=128, truncation=True)
    labels = tokenizer(examples["genz_to_eng_target"], max_length=128, truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_genz_to_eng = augmented_dataset.map(tokenize_genz_to_eng, batched=True)
eval_genz_to_eng = eval_dataset.map(tokenize_genz_to_eng, batched=True)




# --- 5. Training Arguments ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- English → GenZ ---
training_args_eng2genz = TrainingArguments(
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

trainer_eng2genz = Trainer(
    model=model,
    args=training_args_eng2genz,
    train_dataset=train_eng_to_genz,
    eval_dataset=eval_eng_to_genz,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer_eng2genz.train()
trainer_eng2genz.save_model("./models/eng_to_genz")
tokenizer.save_pretrained("./models/eng_to_genz")

# --- GenZ → English ---
model_genz2eng = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer_genz2eng = T5Tokenizer.from_pretrained("t5-small")

training_args_genz2eng = TrainingArguments(
    output_dir="./models/genz_to_eng",
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

trainer_genz2eng = Trainer(
    model=model_genz2eng,
    args=training_args_genz2eng,
    train_dataset=train_genz_to_eng,
    eval_dataset=eval_genz_to_eng,
    tokenizer=tokenizer_genz2eng,
    data_collator=data_collator,
)
trainer_genz2eng.train()
trainer_genz2eng.save_model("./models/genz_to_eng")
tokenizer_genz2eng.save_pretrained("./models/genz_to_eng")

print("✅ Fine-tuning complete! Model saved to ./models/eng_to_genz")

