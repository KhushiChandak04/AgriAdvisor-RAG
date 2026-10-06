from src.i18n import STRINGS, tr


def test_all_languages_have_every_streamlit_ui_string():
    required = {
        "tagline", "caption", "new_chat", "new_chat_title", "your_chats",
        "delete_all", "delete_chat", "safety_note", "try", "placeholder",
        "spinner", "loading", "sources", "page", "listen", "stop", "no_voice",
        "suggested", "kb_error", "kb_hint",
    }
    for language, translations in STRINGS.items():
        assert required <= translations.keys(), (language, required - translations.keys())


def test_tagline_is_available_in_each_supported_language():
    assert tr("English", "tagline")
    assert tr("Hindi", "tagline")
    assert tr("Marathi", "tagline")


def test_key_error_in_fallback_language_has_readable_fallback():
    assert tr("Unknown language", "tagline") == tr("English", "tagline")
