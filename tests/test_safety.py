import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, safety
from src.pdf_processing import chunk_text, clean_text

BLOCKED = [
    "Which pesticide should I use for stem borer in rice?",
    "Recommend a herbicide for weeds in wheat",
    "How much urea per acre for maize?",
    "Calculate NPK dose for cotton per hectare",
    "What should I spray on my soybean leaves?",
    "Best fungicide for sugarcane red rot",
]
ALLOWED = [
    "How to grow soybean in Maharashtra?",
    "What is the seed rate of wheat?",
    "When should I irrigate rice?",
    "What are the symptoms of red rot in sugarcane?",
    "How to control weeds in maize without chemicals?",
    "When to harvest cotton and how to store it?",
    "How many days after planting is the first weeding in maize (DAP)?",
]

def test_blocked():
    for q in BLOCKED:
        assert safety.is_blocked_question(q), q

def test_allowed():
    for q in ALLOWED:
        assert not safety.is_blocked_question(q), q

def test_scrub():
    txt = "Sow in June [1]. Apply 120 kg urea per hectare [2]. Spray imidacloprid at 30 DAP [2]. Harvest at maturity [3]."
    out, mod = safety.scrub_answer(txt)
    assert mod and "urea" not in out and "imidacloprid" not in out
    assert "Sow in June" in out and "Harvest at maturity" in out
    out2, mod2 = safety.scrub_answer("Weed at 30 DAP by hand [1].")
    assert not mod2 and "30 DAP" in out2

def test_crop_detect():
    assert config.detect_crop("FAO_Soyabean_guide") == "soybean"
    assert config.detect_crop("sugar-cane_PoP") == "sugarcane"
    assert config.detect_crop("HR_policy_2023") is None
    assert config.detect_crop("maize") == "maize"

def test_chunking():
    text = " ".join(f"This is sentence number {i} about sowing wheat in November." for i in range(200))
    ch = chunk_text(clean_text(text))
    assert len(ch) > 3 and all(len(c) <= 1200 for c in ch)

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)


def test_devanagari_safety_and_crops():
    assert safety.is_blocked_question("गेहूं में यूरिया कितना डालें")
    assert safety.is_blocked_question("कापसावर फवारणी कोणती करावी")
    assert not safety.is_blocked_question("गहू पेरणी कधी करावी")
    assert not safety.is_blocked_question("खतरा क्या है")
    assert config.detect_crop("उसाची लागवड") == "sugarcane"
    assert config.detect_crop("प्रधान मंत्री") is None
    out, mod = safety.scrub_answer("बुवाई नवंबर में करें। यूरिया 50 किलो डालें।")
    assert mod and "यूरिया" not in out and "बुवाई" in out
