from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_name = "t5-small" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = "Translate English to German: This is a test and e are creating a slang-to-formal translator"
inputs = tokenizer(input_text, return_tensors="pt")

summary_ids = model.generate(inputs.input_ids, max_length=50, num_beams=5, early_stopping=True)

output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("Translation:", output_text)