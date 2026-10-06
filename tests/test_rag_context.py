import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import answer_question


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
            "text": "Rice facts. Sow in June.",
        }][:k]


def test_follow_up_keeps_crop_and_rewrites_query():
    retriever = FakeRetriever()
    history = [{"role": "user", "content": "How should I grow rice?"}]
    with patch("src.rag_pipeline.llm.chat", side_effect=[
            "What factors are important for rice?", "Important factors [1]."]):
        result = answer_question("What factors are important?", retriever, history, crop="rice")

    assert result["crop"] == "rice"
    assert retriever.crop == "rice"
    assert retriever.query == "What factors are important for rice?"


def test_explicit_crop_change_replaces_session_crop():
    retriever = FakeRetriever()
    with patch("src.rag_pipeline.llm.chat", side_effect=[
            "What market needs should be considered for wheat?", "Consider local demand [1]."]):
        result = answer_question(
            "What market needs should be considered for wheat?",
            retriever,
            [{"role": "user", "content": "How should I grow rice?"}],
            crop="rice",
        )

    assert result["crop"] == "wheat"
    assert retriever.crop == "wheat"


def test_unknown_crop_requests_clarification_without_retrieval():
    retriever = FakeRetriever()
    result = answer_question("What factors are important?", retriever)

    assert result["status"] == "needs_clarification"
    assert "Which crop" in result["answer"]
    assert retriever.query is None


def test_selected_language_is_used_via_english_answer_then_translation():
    retriever = FakeRetriever()
    prompts = []

    def fake_chat(messages, **kwargs):
        prompts.append(messages)
        if "translator" in messages[0]["content"]:
            return "- **महत्वपूर्ण बातें:** धान के लिए जरूरी हैं [1]."
        return "- **Factors:** Important for rice [1]."

    with patch("src.rag_pipeline.llm.chat", side_effect=fake_chat):
        result = answer_question("What factors are important for rice?", retriever, language="Hindi")

    assert result["answer"].startswith("- **महत्वपूर्ण बातें:**")
    assert "Hindi" in prompts[-1][0]["content"] and "खरीफ" in prompts[-1][0]["content"]  # glossary


def test_selected_language_localizes_clarification():
    result = answer_question("What factors are important?", FakeRetriever(), language="Marathi")

    assert "कोणत्या पिकाबद्दल" in result["answer"]


# ---------- multilingual / grounding additions ----------
def test_marathi_question_detects_crop_and_is_translated_for_retrieval():
    retriever = FakeRetriever()
    with patch("src.rag_pipeline.llm.chat", side_effect=[
            "When to sow wheat?", "- **Sowing:** November [1].", "- **पेरणी:** नोव्हेंबर [1]."]):
        result = answer_question("गव्हाची पेरणी कधी करावी?", retriever, language="Marathi")
    assert result["crop"] == "wheat"
    assert retriever.query == "When to sow wheat?"      # English query used for retrieval


def test_hindi_chemical_request_is_blocked_without_llm_call():
    with patch("src.rag_pipeline.llm.chat", side_effect=AssertionError("LLM must not be called")):
        result = answer_question("धान के लिए कौन सा कीटनाशक डालें", FakeRetriever(), language="Hindi")
    assert result["status"] == "blocked"


def test_followup_chemical_request_blocked_after_rewrite():
    with patch("src.rag_pipeline.llm.chat", return_value="How much urea per acre for rice?"):
        result = answer_question("and how much of it?", FakeRetriever(),
                                 [{"role": "user", "content": "fertilizer for rice"}], crop="rice")
    assert result["status"] == "blocked"


def test_rewrite_failure_falls_back_instead_of_crashing():
    retriever = FakeRetriever()
    calls = {"n": 0}

    def flaky(messages, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("empty answer")
        return "Sow in June [1]."
    with patch("src.rag_pipeline.llm.chat", side_effect=flaky):
        result = answer_question("When to sow?", retriever,
                                 [{"role": "user", "content": "How to grow rice?"}], crop="rice")
    assert result["status"] in ("ok", "ungrounded_trimmed", "no_context")
    assert retriever.crop == "rice"


def test_invented_numbers_are_removed():
    class R(FakeRetriever):
        def search(self, query, k, crop):
            return [{"score": 0.9, "source": "r.pdf", "page": 1, "crop": crop,
                     "text": "Sow rice in June at 20 cm spacing and 2 cm depth."}]
    with patch("src.rag_pipeline.llm.chat",
               return_value="- **Season:** June [1].\n- **Spacing:** 20 cm [1].\n- **Yield:** 99 quintals [1]."):
        result = answer_question("rice sowing", R())
    assert "99" not in result["answer"] and "20 cm" in result["answer"]


def test_all_numbers_invented_gives_not_found():
    with patch("src.rag_pipeline.llm.chat", return_value="- **Yield:** 77 quintals per acre [1]."):
        result = answer_question("rice yield", FakeRetriever())
    assert result["status"] == "no_context"


def test_weak_passages_are_dropped():
    class R(FakeRetriever):
        def search(self, query, k, crop):
            return [{"score": 0.6, "source": "a", "page": 1, "crop": crop, "text": "strong text"},
                    {"score": 0.1, "source": "b", "page": 2, "crop": crop, "text": "weak text"}]
    seen = {}

    def fake(messages, **kw):
        seen["p"] = messages[1]["content"]
        return "Strong fact [1]."
    with patch("src.rag_pipeline.llm.chat", side_effect=fake):
        answer_question("rice facts", R())
    assert "strong text" in seen["p"] and "weak text" not in seen["p"]


# ---------- fixes from field test transcript ----------
class MultiRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k, crop):
        self.calls.append(crop)
        return [{"score": 0.7, "source": f"{crop}.pdf", "page": 1, "crop": crop,
                 "text": f"{crop} land preparation: plough once. Sow in June."}]


def test_comparison_retrieves_every_crop():
    r = MultiRetriever()
    seen = {}

    def fake(messages, **kw):
        if "fact checker" in messages[0]["content"]:
            return "1: YES"
        seen["p"] = messages[1]["content"]
        return "- **Land preparation:** Rice: plough once [1]; Wheat: plough once [2]."
    with patch("src.rag_pipeline.llm.chat", side_effect=fake):
        res = answer_question("Compare the cultivation practices of rice and wheat from land preparation", r)
    assert r.calls == ["rice", "wheat"] and res["crop"] == "rice,wheat"
    assert "comparison of: rice, wheat" in seen["p"]


def test_off_topic_and_price_questions_have_own_replies():
    hist = [{"role": "user", "content": "How to grow rice?"}]
    with patch("src.rag_pipeline.llm.chat", side_effect=AssertionError("no LLM")):
        assert answer_question("Who is the Prime Minister of India?", FakeRetriever(), hist,
                               crop="rice")["status"] == "out_of_scope"
        assert answer_question("What is the price of rice today?", FakeRetriever())["status"] == "price"


def test_devanagari_question_gets_answer_in_that_language_even_if_ui_is_english():
    r = FakeRetriever()

    def fake(messages, **kw):
        sysmsg = messages[0]["content"]
        if "Rewrite" in sysmsg:
            return "When to sow rice?"
        if "translator" in sysmsg:
            return "- **पेरणी:** जून महिन्यात पेरणी करा [1]."
        return "- **Sowing:** Sow rice in the month of June [1]."
    with patch("src.rag_pipeline.llm.chat", side_effect=fake):
        res = answer_question("भाताची लागवड कशी करावी?", r, language="English")
    assert res["language"] == "Marathi" and r.crop == "rice" and "पेरणी" in res["answer"]


def test_unknown_conditions_are_not_listed_unless_requested_and_stray_tags_removed():
    seen = {}

    def fake(messages, **kw):
        if "fact checker" in messages[0]["content"]:
            return "1: YES"
        seen["p"] = messages[1]["content"]
        return "- **Water:** Keep shallow water [recent] [1]."
    with patch("src.rag_pipeline.llm.chat", side_effect=fake):
        res = answer_question("rice in heavy rainfall, clayey soil and poor drainage", FakeRetriever())
    assert "do NOT mention" not in seen["p"]
    assert "Not covered" not in res["answer"]
    assert "[recent]" not in res["answer"]


# ---------- fact-check, translation safety, comparison ----------
class TextRetriever(FakeRetriever):
    def search(self, query, k, crop):
        return [{"score": 0.8, "source": "r.pdf", "page": 1, "crop": crop,
                 "text": "Plough twice and harrow once. Level the field. Weeds cannot germinate under water. Sow in June."}]


def test_fact_checker_removes_invented_advice_without_unsolicited_missing_list():
    def fake(messages, **kw):
        sysmsg = messages[0]["content"]
        if "fact checker" in sysmsg:
            return "1: YES\n2: NO\n3: NO"
        return ("- **Land preparation:** Plough twice and harrow once [1].\n"
                "- **Drainage:** Install drainage lines across the field [1].\n"
                "- **Timing:** Plant in the first fortnight of April [1].")
    with patch("src.rag_pipeline.llm.chat", side_effect=fake):
        res = answer_question("I want to grow rice with heavy rainfall, clayey soil and poor drainage. "
                              "What practices should I follow to manage these conditions?", TextRetriever())
    assert "Plough twice" in res["answer"]
    assert "drainage lines" not in res["answer"] and "April" not in res["answer"]
    assert "Not covered" not in res["answer"]


def test_translation_that_adds_numbers_is_rejected_instead_of_falling_back_to_english():
    def fake(messages, **kw):
        if "translator" in messages[0]["content"]:
            return "- **पेरणी:** जून में 45 दिन [1]."
        return "- **Sowing:** Sow in June [1]."
    with patch("src.rag_pipeline.llm.chat", side_effect=fake):
        try:
            answer_question("When to sow rice?", TextRetriever(), language="Hindi")
        except RuntimeError as exc:
            assert "did not preserve" in str(exc)
        else:
            raise AssertionError("An unsafe translation must not be returned in English")


def test_comparison_is_one_merged_answer_not_per_crop_refusals():
    calls = {"gen": 0}

    def fake(messages, **kw):
        sysmsg = messages[0]["content"]
        if "fact checker" in sysmsg:
            return "1: YES"
        calls["gen"] += 1
        return "- **Land preparation:** Rice: plough twice [1]; Wheat: plough twice [2]."
    with patch("src.rag_pipeline.llm.chat", side_effect=fake):
        res = answer_question("Compare rice and wheat land preparation", MultiRetriever())
    assert calls["gen"] == 1 and "Rice:" in res["answer"] and "Wheat:" in res["answer"]


# ---------- round 3: months, comparison wording, scrub note ----------
def test_invented_month_is_dropped():
    with patch("src.rag_pipeline.llm.chat",
               return_value="- **Sowing:** Sow in June [1].\n- **Timing:** Plant in the first fortnight of April for Kharif [1]."):
        res = answer_question("rice sowing time", FakeRetriever())
    assert "June" in res["answer"] and "April" not in res["answer"]


def test_comparison_prompt_forbids_refusal_and_asks_one_bullet_per_aspect():
    r, seen = MultiRetriever(), {}

    def fake(messages, **kw):
        if "fact checker" in messages[0]["content"]:
            return "1: YES"
        seen["p"] = messages[1]["content"]
        return "- **Land preparation:** Rice: plough once [1]; Wheat: plough once [2]."
    with patch("src.rag_pipeline.llm.chat", side_effect=fake):
        answer_question("Compare the cultivation practices of rice and wheat from land preparation", r)
    assert "ONE bullet per aspect" in seen["p"] and "Never apologise" in seen["p"]


def test_chemical_note_only_when_farmer_asked_about_inputs():
    reply = "- **Sowing:** Sow in June [1].\n- **Inputs:** Apply urea 50 kg [1]."
    with patch("src.rag_pipeline.llm.chat", return_value=reply):
        plain = answer_question("When to sow rice?", FakeRetriever())
        asked = answer_question("Tell me about fertilizer management in rice", FakeRetriever())
    assert "KVK" not in plain["answer"] and "urea" not in plain["answer"]
    assert "urea" not in asked["answer"]
