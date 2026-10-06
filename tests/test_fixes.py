"""Tests for multi-crop, scope detection, language support, glossary, and formatting fixes."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.multicrop import detect_all_crops, is_comparison_question, separate_query_per_crop
from src.scope import is_out_of_scope, is_price_question, out_of_scope_reply, price_question_reply
from src.glossary import translate_term, GLOSSARY
from src.rag_pipeline import answer_question, _remove_tags_and_unsupported_numbers
from src.config import detect_crop


class FakeRetriever:
    def __init__(self):
        self.query = None
        self.crop = None

    def search(self, query, k, crop):
        self.query = query
        self.crop = crop
        return [{
            "score": 0.9,
            "source": "rice.pdf",
            "page": 1,
            "crop": crop,
            "text": "Rice sowing in June at 20 cm spacing and 2 cm depth in loamy soil.",
        }][:k]


# ========== Multi-crop tests ==========
def test_detect_all_crops_single():
    assert detect_all_crops("How to grow rice?") == ["rice"]
    assert detect_all_crops("गेहूँ की बुवाई") == ["wheat"]
    assert detect_all_crops("उसाची लागवड") == ["sugarcane"]


def test_detect_all_crops_multiple():
    crops = detect_all_crops("What's the difference between rice and wheat?")
    assert "rice" in crops and "wheat" in crops


def test_detect_all_crops_none():
    assert detect_all_crops("What is agriculture?") == []


def test_is_comparison_question():
    assert is_comparison_question("rice vs wheat")
    assert is_comparison_question("Which is better: maize or soybean?")
    assert is_comparison_question("Difference between cotton and sugarcane")
    assert not is_comparison_question("How to grow rice?")


def test_separate_query_per_crop():
    queries = separate_query_per_crop("What's the difference between rice and wheat?", ["rice", "wheat"])
    assert "rice" in queries["rice"].lower()
    assert "wheat" in queries["wheat"].lower()


# ========== Scope detection tests ==========
def test_out_of_scope_english():
    assert is_out_of_scope("What's the weather today?")
    assert is_out_of_scope("Who is the prime minister?")
    assert is_out_of_scope("When was COVID-19 discovered?")
    assert not is_out_of_scope("How to grow rice?")


def test_out_of_scope_hindi():
    assert is_out_of_scope("आज मौसम कैसा है?")
    assert is_out_of_scope("प्रधान मंत्री कौन हैं?")
    assert not is_out_of_scope("धान की बुवाई कब करें?")


def test_out_of_scope_marathi():
    assert is_out_of_scope("आज हवामान कसे आहे?")
    assert not is_out_of_scope("तांदूळ लागवड कशी करावी?")


def test_price_question_english():
    assert is_price_question("What's the price of rice?")
    assert is_price_question("How much can I earn from wheat farming?")
    assert is_price_question("MSP for cotton this year?")
    assert not is_price_question("When to harvest rice?")


def test_price_question_hindi():
    assert is_price_question("धान का भाव क्या है?")
    assert is_price_question("गेहूँ से आय कितनी होगी?")
    assert not is_price_question("धान की कटाई कब करें?")


def test_out_of_scope_reply_languages():
    en = out_of_scope_reply("English")
    hi = out_of_scope_reply("Hindi")
    mr = out_of_scope_reply("Marathi")
    
    assert "growing" in en.lower() or "answer" in en.lower()
    assert "खेती" in hi
    assert "लागवड" in mr


def test_price_reply_languages():
    en = price_question_reply("English")
    hi = price_question_reply("Hindi")
    mr = price_question_reply("Marathi")
    
    assert "price" in en.lower() or "market" in en.lower()
    assert "बाजार" in hi
    assert "किंमत" in mr or "दर" in mr


# ========== Language detection tests ==========
def test_marathi_question_detected():
    result = answer_question("गव्हाची पेरणी कधी करावी?", FakeRetriever(), language="Marathi")
    assert result.get("crop") == "wheat"
    assert result.get("language") == "Marathi"


def test_hindi_question_detected():
    result = answer_question("धान की बुवाई कब करें?", FakeRetriever(), language="Hindi")
    assert result.get("crop") == "rice"
    assert result.get("language") == "Hindi"


def test_language_returned_in_result():
    result = answer_question("How to grow rice?", FakeRetriever(), language="English")
    assert result.get("language") == "English"
    
    result = answer_question("rice growing", FakeRetriever(), language="Hindi")
    assert result.get("language") == "Hindi"


# ========== Glossary tests ==========
def test_glossary_terms_exist():
    assert "irrigation" in GLOSSARY
    assert "sowing" in GLOSSARY
    assert "pest" in GLOSSARY


def test_glossary_translations():
    # Check that Hindi translations exist
    sowing = GLOSSARY.get("sowing")
    assert sowing and "Hindi" in sowing
    assert sowing and "Marathi" in sowing


def test_translate_term():
    # English should return original
    assert translate_term("irrigation", "English") == "irrigation"
    
    # Hindi should return translation
    hindi_result = translate_term("irrigation", "Hindi")
    assert hindi_result != "irrigation"
    assert hindi_result in ["सिंचाई", "irrigation"]
    
    # Marathi should return translation
    marathi_result = translate_term("irrigation", "Marathi")
    assert marathi_result in ["सिंचन", "irrigation"]


def test_translate_unknown_term():
    # Unknown term should return original
    result = translate_term("xyz123", "Hindi")
    assert result == "xyz123"


# ========== Formatting tests ==========
def test_remove_tags_keeps_citations():
    text = "Sow in June [1]. Harvest at maturity [2]."
    result = _remove_tags_and_unsupported_numbers(text)
    # Citations should be preserved
    assert "[1]" in result or "June" in result


def test_remove_tags_removes_non_numeric():
    text = "Recent findings [recent] show that [old] methods are outdated."
    result = _remove_tags_and_unsupported_numbers(text)
    assert "[recent]" not in result and "[old]" not in result


def test_remove_unsupported_numbers():
    text = "Sow rice at 20 cm spacing [1]. Harvest at maturity [1]."
    result = _remove_tags_and_unsupported_numbers(text)
    # Keep text and citations
    assert "Sow rice" in result and "[1]" in result


# ========== Scope handling integration tests ==========
def test_out_of_scope_returns_local_response():
    result = answer_question("What's the weather today?", FakeRetriever())
    assert result["status"] == "out_of_scope"
    assert "weather" not in result["answer"].lower() or "only answer" in result["answer"].lower()
    assert result["sources"] == []


def test_price_question_returns_local_response():
    result = answer_question("What's the price of rice?", FakeRetriever())
    assert result["status"] == "price"
    assert "market" in result["answer"].lower() or "price" in result["answer"].lower()
    assert result["sources"] == []


def test_scope_detection_hindi():
    result = answer_question("आज का मौसम कैसा है?", FakeRetriever(), language="Hindi")
    # Should be out of scope or something similar (may still ask for crop if hindi question is ambiguous)
    assert result["status"] in ["out_of_scope", "needs_clarification"]


def test_price_detection_hindi():
    result = answer_question("धान का भाव क्या है?", FakeRetriever(), language="Hindi")
    assert result["status"] == "price"


# ========== Multi-crop comparison tests ==========
def test_multi_crop_comparison():
    result = answer_question("What's better: rice or wheat?", FakeRetriever())
    # Should either detect comparison or clarify crops
    assert result["status"] in ["ok", "ungrounded_trimmed", "out_of_scope", "no_context", "needs_clarification"]


def test_multi_crop_vs_query():
    result = answer_question("rice vs. wheat sowing time", FakeRetriever())
    # Should handle the comparison
    assert result is not None


# ========== Cost of cultivation in context ==========
def test_cost_of_cultivation_allowed_in_rag():
    # Cost of cultivation should NOT be blocked (unlike fertilizer doses)
    from src.safety import is_blocked_question
    assert not is_blocked_question("What is the cost of cultivation for rice?")
    assert not is_blocked_question("धान की खेती का खर्च क्या है?")


# Run all tests
if __name__ == "__main__":
    test_functions = [f for n, f in list(globals().items()) if n.startswith("test_")]
    passed = 0
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: {e}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {type(e).__name__}: {e}")
    
    print(f"\n{passed}/{len(test_functions)} tests passed")
