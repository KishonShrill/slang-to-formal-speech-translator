import csv
import time
import os
import google.generativeai as genai
from tqdm import tqdm # A nice progress bar
from dotenv import load_dotenv
from datasets import load_dataset # <-- Import this

# --- 1. CONFIGURE YOUR API KEY ---
load_dotenv() # <-- Load the .env file
API_KEY = os.getenv("GEMINI_API_KEY") # <-- Get key from environment

if not API_KEY:
    print("Error: 'GEMINI_API_KEY' not found in .env file or environment variables.")
    print("Please create a .env file and add: GEMINI_API_KEY=your_key_here")
    exit()

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring API key: {e}")
    exit()

# --- 1b. TEST SLICING CONFIGURATION ---
USE_TEST_SLICE = True  # Set to False to run on all data
TEST_SLICE_SIZE = 2      # Number of samples to process if USE_TEST_SLICE is True

# --- 2. LOAD YOUR HUGGING FACE DATASET ---
print("Loading 'MLBtrio/genz-slang-dataset' from Hugging Face...")
try:
    # Load the dataset (it only has a 'train' split)
    raw_dataset = load_dataset("MLBtrio/genz-slang-dataset", split='train')
    print(f"Dataset loaded. Total examples: {len(raw_dataset)}")
    
    # Filter out any rows where the 'Example' is missing, just in case
    raw_dataset = raw_dataset.filter(lambda x: x['Example'] is not None and len(x['Example']) > 0)
    print(f"Filtered to {len(raw_dataset)} examples with valid 'Example' fields.")

    # Show a sample
    print("\nDataset sample:")
    print(raw_dataset[0])
    
except Exception as e:
    print(f"Error loading Hugging Face dataset: {e}")
    print("Please make sure you have 'pip install datasets' and an internet connection.")
    exit()


# --- 3. SETUP THE MODEL AND PROMPT ---
# We'll use a fast and powerful model
model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')

# This is the "instruction" for the model
# We use "few-shot" prompting by giving it one perfect example.
SYSTEM_PROMPT = """
You are a "Style Translator." Your job is to translate a Gen Z slang sentence into its equivalent in formal, standard English.
- Do NOT explain the translation.
- Do NOT add quotation marks.
- Only output the formal English translation.

Here is an example:

Gen Z: He’s savage on the court.
Formal: He is a fierce and bold competitor on the court.
"""

def get_formal_translation(genz_sentence):
    """
    Calls the Gemini API to get a formal translation for a single sentence.
    """
    try:
        # We combine the main prompt with the specific user request
        full_prompt = f"{SYSTEM_PROMPT}\nGen Z: {genz_sentence}\nFormal:"
        
        response = model.generate_content(
            full_prompt,
            # We set a low temperature for less "creative" and more direct translations
            generation_config=genai.types.GenerationConfig(
                temperature=0.1
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"\nError translating '{genz_sentence}': {e}")
        return None

# --- 4. MAIN EXECUTION: Loop, Translate, and Save ---
output_filename = "genz_to_formal_dataset.csv"
parallel_data = []

# --- Apply test slice if enabled ---
if USE_TEST_SLICE:
    print(f"--- TEST MODE ON: Slicing to first {TEST_SLICE_SIZE} samples. ---")
    # We use .select() for Hugging Face datasets, which is like slicing
    data_to_process = raw_dataset.select(range(TEST_SLICE_SIZE))
else:
    print("--- TEST MODE OFF: Processing all samples. ---")
    data_to_process = raw_dataset

total_items = len(data_to_process) # <-- Get total count
print(f"Starting dataset generation... will process {total_items} examples.")
print(f"Output will be saved to: {output_filename}")

# Use tqdm for a progress bar
# We will use enumerate and wrap data_to_process with tqdm
for i, item in enumerate(tqdm(data_to_process, desc="Overall Progress")):
    # --- Use the dataset column names ---
    genz_example = item["Example"] 
    slang_term = item["Slang"]
    items_left = total_items - (i + 1) # <-- Calculate items left

    # --- NEW LOGGER ---
    # Use tqdm.write() to print log messages without messing up the progress bar
    tqdm.write(f"\n[Item {i+1}/{total_items}] Translating '{slang_term}': \"{genz_example}\" | {items_left} items remaining.")
    
    # Call our API function
    formal_example = get_formal_translation(genz_example)
    
    if formal_example:
        # Add the new pair to our list
        parallel_data.append({
            "genz_sentence": genz_example,
            "formal_sentence": formal_example
        })
    
    # --- IMPORTANT: Rate Limiting ---
    # Add a delay to avoid hitting the free-tier limit (e.g., 60 requests/min)
    time.sleep(1) 

print("\nTranslation complete!")

# --- 5. WRITE TO CSV ---
try:
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        # We need two columns: 'genz_sentence' and 'formal_sentence'
        # (Or you could reverse them for your other model)
        writer = csv.DictWriter(f, fieldnames=["genz_sentence", "formal_sentence"])
        writer.writeheader()
        writer.writerows(parallel_data)
    
    print(f"Successfully saved {len(parallel_data)} pairs to {output_filename}")
    print("\nFile Content Preview:")
    for i, row in enumerate(parallel_data[:5]):
        print(f"  Row {i+1}:")
        print(f"    Gen Z: {row['genz_sentence']}")
        print(f"    Formal: {row['formal_sentence']}")

except Exception as e:
    print(f"Error writing to CSV: {e}")
