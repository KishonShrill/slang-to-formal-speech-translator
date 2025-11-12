import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- Optional: cache loaded model globally to avoid reloading ---
_model_cache = {
    "eng_to_genz": None,
    "genz_to_eng": None
}


def load_model(direction: str):
    """
    Loads and caches the fine-tuned T5 model for the given direction.
    Example directions: 'eng_to_genz', 'genz_to_eng'
    """
    global _model_cache

    if _model_cache[direction] is not None:
        return _model_cache[direction]

    model_path = f"models/{direction}"  # your fine-tuned model folder
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} 💽")
    model.to(device)

    _model_cache[direction] = (tokenizer, model, device)
    return tokenizer, model, device


def translate_text(text: str, direction: str) -> str:
    """
    Translate text using the fine-tuned T5 model.
    direction: "eng_to_genz" or "genz_to_eng"
    """
    tokenizer, model, device = load_model(direction)

    # Add a prefix like T5 uses for translation tasks
    prefix = "translate English to GenZ: " if direction == "eng_to_genz" else "translate GenZ to English: "
    input_text = prefix + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated.strip()

