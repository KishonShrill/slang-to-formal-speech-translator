def translate_text(text: str, direction: str) -> str:
    """
    Placeholder translation logic.
    Later replace with a transformer-based model.
    """
    ENG_TO_GENZ = {
        "hello": "yo",
        "good": "slaps",
        "great": "slaps",
        "bad": "mid",
        "friend": "bestie",
        "i understand": "bet",
        "i agree": "bet",
        "no": "nah",
        "yes": "bet",
        "that's funny": "i'm weak",
        "opinion": "take",
        "style": "drip",
        "food": "zaza",
        "fake": "cap",
        "lying": "capping"
    }

    GENZ_TO_ENG = {
        "yo": "hello",
        "slaps": "is very good",
        "mid": "mediocre / bad",
        "bestie": "friend",
        "bet": "i understand / i agree / yes",
        "nah": "no",
        "i'm weak": "that's funny",
        "take": "opinion",
        "drip": "style",
        "zaza": "food",
        "cap": "a lie / fake",
        "capping": "lying"
    }

    lookup_dict = ENG_TO_GENZ if direction == "eng_to_genz" else GENZ_TO_ENG

    words = text.split()
    result = []

    for word in words:
        clean = word.lower().strip(".,!?")
        result.append(lookup_dict.get(clean, word))

    return " ".join(result)

