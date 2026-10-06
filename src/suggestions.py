"""Localized starter questions shown before a chat has messages."""

_SUGGESTIONS = {
    "English": [
        "When should rice be sown?",
        "What soil is suitable for wheat?",
        "How should maize be harvested?",
        "How should soybean be stored?",
    ],
    "Hindi": [
        "धान की बुवाई कब करनी चाहिए?",
        "गेहूँ के लिए कौन सी मिट्टी उपयुक्त है?",
        "मक्का की कटाई कैसे करें?",
        "सोयाबीन का भंडारण कैसे करें?",
    ],
    "Marathi": [
        "भाताची पेरणी कधी करावी?",
        "गव्हासाठी कोणती माती योग्य आहे?",
        "मक्याची कापणी कशी करावी?",
        "सोयाबीनची साठवण कशी करावी?",
    ],
}


def suggest(history, language: str):
    """Return localized, document-grounded starter prompts."""
    return list(_SUGGESTIONS.get(language, _SUGGESTIONS["English"]))
