"""Regression tests for multi-crop retrieval, language memory and chunk metadata."""
from src.multicrop import detect_all_crops
from src.pdf_processing import chunk_page
from src.rag_agent import run_rag_agent
from src.rag_pipeline import _numbers, _translate_labels, answer_question, smalltalk_reply


class CropRetriever:
    def __init__(self, missing=()):
        self.calls = []
        self.missing = set(missing)

    def search(self, query, k, crop):
        self.calls.append((query, crop))
        if crop in self.missing:
            return []
        return [{
            "score": 0.9,
            "source": f"{crop}.pdf",
            "page": 4,
            "section": "Sowing",
            "crop": crop,
            "text": f"{crop} sowing should use the locally suitable season.",
        }][:k]


def test_detects_all_six_crops_in_question_order():
    assert detect_all_crops("Rice, wheat, maize, cotton, soybean and sugarcane") == [
        "rice", "wheat", "maize", "cotton", "soybean", "sugarcane"
    ]


def test_greeting_with_self_introduction_is_treated_as_greeting():
    assert smalltalk_reply("Hi, I am xyz") is not None
    assert smalltalk_reply("Hello, my name is Jane Doe") is not None


def test_answer_flow_keeps_introduction_greeting_local():
    result = answer_question("Hi, I am xyz", CropRetriever())
    assert result["status"] == "smalltalk"


def test_agent_uses_crop_specific_subqueries_and_metadata():
    retriever = CropRetriever()
    state = run_rag_agent(
        "Compare sowing time for rice and wheat",
        "Compare sowing time for rice and wheat",
        retriever,
        ["rice", "wheat"],
        "",
        5,
    )
    queries = dict((crop, query.lower()) for query, crop in retriever.calls)
    assert "rice" in queries["rice"] and "wheat" not in queries["rice"]
    assert "wheat" in queries["wheat"] and "rice" not in queries["wheat"]
    assert {p["crop"] for p in state["passages"]} == {"rice", "wheat"}
    assert all(p["page"] == 4 and p["section"] == "Sowing" for p in state["passages"])


def test_comparison_keeps_each_crop_in_cited_sources():
    retriever = CropRetriever()
    result = answer_question("Compare rice and wheat sowing time", retriever)
    assert result["crop"] == "rice,wheat"
    assert {source["crop"] for source in result["sources"]} == {"rice", "wheat"}
    assert all(source["section"] == "Sowing" for source in result["sources"])


def test_multi_question_query_retrieves_and_answers_each_requested_aspect(monkeypatch):
    class SoybeanRetriever:
        def __init__(self):
            self.calls = []

        def search(self, query, k, crop):
            self.calls.append(query.lower())
            if "soil" in query.lower():
                text, section = "Soybean needs well-drained soil.", "Soil"
            else:
                text, section = "Soybean cultivation includes seed selection and sowing.", "Cultivation"
            return [{
                "score": 0.9,
                "source": "soybean.pdf",
                "page": 2,
                "section": section,
                "crop": crop,
                "text": text,
            }]

    prompts = []

    def answer(messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[1]["content"]
        if "rewrite the farmer" in system:
            return (
                "Best soil type for soybean cultivation."
                if "soil" in user.lower()
                else "How to grow soybean?"
            )
        if "fact checker" in system:
            return "1: YES\n2: YES"
        prompts.append(user)
        return (
            "- **Cultivation:** The guide discusses seed selection and sowing [1].\n"
            "- **Soil:** Soybean needs well-drained soil [2]."
        )

    monkeypatch.setattr("src.rag_pipeline.llm.chat", answer)
    retriever = SoybeanRetriever()
    result = answer_question(
        "how to grow soyabean? Which type of soil is best suitable for it",
        retriever,
        history=[{"role": "user", "content": "hiii"}],
    )

    assert len(retriever.calls) == 2
    assert any("grow" in query for query in retriever.calls)
    assert any("soil" in query for query in retriever.calls)
    assert "how to grow soybean" in prompts[0].lower()
    assert "soil" in prompts[0].lower()
    assert "address each one in its own bullet" in prompts[0].lower()
    assert "Cultivation" in result["answer"]
    assert "Soil" in result["answer"]
    assert result["crop"] == "soybean"
    assert {source["section"] for source in result["sources"]} == {"Cultivation", "Soil"}


def test_language_is_remembered_for_english_followup():
    result = answer_question(
        "And what about storage?",
        CropRetriever(),
        history=[{"role": "user", "content": "गव्हाची पेरणी कधी करावी?"}],
        crop="wheat",
        language="English",
        prev_language="Marathi",
    )
    assert result["language"] == "Marathi"
    assert any("\u0900" <= char <= "\u097f" for char in result["answer"])


def test_previous_turns_are_passed_into_followup_rewrite(monkeypatch):
    captured = []

    def rewrite_and_answer(messages, **kwargs):
        captured.append(messages)
        system = messages[0]["content"].lower()
        if "rewrite the farmer" in system:
            return "How to store rice?"
        if "fact checker" in system:
            return "1: YES"
        return "- **Storage:** Use the method described in the rice document [1]."

    monkeypatch.setattr("src.rag_pipeline.llm.chat", rewrite_and_answer)
    result = answer_question(
        "And what about storage?",
        CropRetriever(),
        history=[
            {"role": "user", "content": "How should I grow rice?"},
            {"role": "assistant", "content": "The guide discusses rice cultivation."},
        ],
        crop="rice",
    )
    assert result["status"] == "ok"
    rewrite_prompt = captured[0][1]["content"]
    assert "How should I grow rice?" in rewrite_prompt
    assert "The guide discusses rice cultivation." in rewrite_prompt


def test_explicit_language_choice_overrides_detected_input_language():
    result = answer_question(
        "धान की बुवाई कब करें?",
        CropRetriever(),
        language="English",
        force_language=True,
    )
    assert result["language"] == "English"


def test_page_chunks_retain_section_heading():
    text = (
        "Soil and Climate\n"
        "Rice grows best in suitable soil conditions. The documents describe the climate and "
        "soil context for crop establishment and its seasonal conditions.\n"
        "Sowing Time\n"
        "Sowing should follow the locally suitable season. The section explains the planting "
        "time and field preparation before establishing the crop."
    )
    chunks = chunk_page(text)
    assert {chunk["section"] for chunk in chunks} == {"Soil and Climate", "Sowing Time"}
    assert all(chunk["section"] in chunk["text"] for chunk in chunks)


def test_comparison_with_missing_crop_evidence_keeps_available_crop():
    retriever = CropRetriever(missing={"wheat"})
    result = answer_question("Compare rice and wheat sowing time", retriever)
    assert result["status"] in {"ok", "ungrounded_trimmed"}
    assert {source["crop"] for source in result["sources"]} == {"rice"}


def test_broad_question_does_not_list_unsolicited_missing_details(monkeypatch):
    prompts = []

    def answer(messages, **kwargs):
        prompts.append(messages[1]["content"])
        return "- **Cultivation:** Follow the supported practices in the crop guide [1]."

    monkeypatch.setattr("src.rag_pipeline.llm.chat", answer)
    result = answer_question("How should I grow rice?", CropRetriever())
    assert "Not covered" not in result["answer"]
    assert "documents do NOT mention" not in prompts[0]
    assert "Not covered:**" not in prompts[0]


def test_english_bullet_labels_are_translated_for_hindi_and_marathi():
    answer = (
        "- **High-quality seed:** साफ बियाणे [1].\n"
        "- **Land preparation:** शेत तयार करा [2].\n"
        "- **Transplanting:** रोपे लावा [3]."
    )
    hindi = _translate_labels(answer, "Hindi")
    marathi = _translate_labels(answer, "Marathi")

    assert "**उच्च गुणवत्ता वाला बीज:**" in hindi
    assert "**भूमि की तैयारी:**" in hindi
    assert "**रोपाई:**" in hindi
    assert "**उत्तम प्रतीचे बियाणे:**" in marathi
    assert "**जमिनीची पूर्वमशागत:**" in marathi
    assert "**पुनर्लागवड:**" in marathi
    assert "[1]" in hindi and "[2]" in marathi


def test_devanagari_month_matching_does_not_treat_common_words_as_may():
    assert _numbers("मिट्टी मेरी उपयुक्त है") == set()
    assert _numbers("मिट्टी में पानी रुकता है") == set()
    assert _numbers("सोयाबीन की बुवाई मई में करें") == {"may"}
    assert _numbers("Sow in May") == {"may"}


def test_hindi_soil_answer_without_quantities_does_not_fail_translation(monkeypatch):
    class SoilRetriever:
        def search(self, query, k, crop):
            return [{
                "score": 0.9,
                "source": "soybean.pdf",
                "page": 3,
                "section": "Soil",
                "crop": crop,
                "text": "Soybean grows well in well-drained sandy loam soil.",
            }]

    def fake_chat(messages, **kwargs):
        system = messages[0]["content"].lower()
        if "rewrite the farmer" in system:
            return "Which soil is suitable for soybean?"
        if "translator" in system:
            return "- **मिट्टी:** सोयाबीन के लिए अच्छी जल निकासी वाली बलुई दोमट मिट्टी उपयुक्त है [1]."
        return "- **Soil:** Soybean grows well in well-drained sandy loam soil [1]."

    monkeypatch.setattr("src.rag_pipeline.llm.chat", fake_chat)
    result = answer_question(
        "सोयाबीन के लिए कौन सी मिट्टी उपयुक्त है?",
        SoilRetriever(),
        language="Hindi",
    )

    assert result["status"] == "ok"
    assert "सोयाबीन के लिए" in result["answer"]
    assert "did not preserve" not in result["answer"]
