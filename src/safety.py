"""Safety guardrails: NO pesticide/herbicide/fungicide advice, NO fertilizer calculations."""
import re
from langsmith import traceable

from . import config

# Chemical crop-protection terms
CHEMICAL_TERMS = (
    r"pesticides?|insecticides?|herbicides?|fungicides?|weedicides?|nematicides?|acaricides?|"
    r"miticides?|rodenticides?|agro-?chemicals?|chemical (spray|control)|spray(ing)? schedule|"
    r"glyphosate|atrazine|pendimethalin|chlorpyrifos|imidacloprid|cypermethrin|mancozeb|"
    r"carbendazim|2,?4-?d|paraquat|endosulfan|monocrotophos|thiamethoxam|"
    r"emamectin|profenofos|acephate|butachlor|metribuzin"
)
FERTILIZER_TERMS = (r"fertili[sz]ers?|urea|npk|nitrogen|phosphorus|potash|di-?ammonium phosphate|"
                    r"muriate of potash|super ?phosphate|ammonium sulphate|manure dose")
CALC_WORDS = r"calculate|calculation|how (much|many)|dose|dosage|rate|quantity|amount|schedule|per (acre|hectare|ha)|kg/?ha|bags?|recommend"

_CHEM_RE = re.compile(rf"\b({CHEMICAL_TERMS})\b", re.I)
_FERT_RE = re.compile(rf"\b({FERTILIZER_TERMS})\b", re.I)
_CALC_RE = re.compile(rf"\b({CALC_WORDS})\b", re.I)
_RECOMMEND_PEST_RE = re.compile(r"\b(spray|spraying|dusting)\b", re.I)

# Hindi / Marathi (Devanagari: no \b, use explicit boundaries)
_NB = r"(?![\u0900-\u097F])"
_NA = r"(?<![\u0900-\u097F])"
_DEV_CHEM = re.compile(
    r"कीटनाशक|कीटकनाशक|खरपतवारनाशक|तणनाशक|फफूंदनाशक|फफूंदीनाशक|कवकनाशक|बुरशीनाशक|कीटनाशी|"
    r"रासायनिक\s*(छिड़काव|फवारणी|दवा|औषध)|फवारणी|छिड़काव|स्प्रे|ग्लायफोसेट|ग्लाइफोसेट|मॅन्कोझेब|मैन्कोज़ेब")
_DEV_FERT = re.compile(
    r"उर्वरक|रासायनिक\s*खाद|युरिया|यूरिया|डीएपी|एनपीके|पोटाश|नत्र|नायट्रोजन|नाइट्रोजन|फॉस्फरस|फास्फोरस|स्फुरद|पालाश|"
    + _NA + r"खाद(?:ें|ों)?" + _NB + "|" + _NA + r"खत(?:े|ां|ाचा|ाची|ाचे|ाचं|ासाठी|ांचा|ांची|ांचे)?" + _NB)
_DEV_CALC = re.compile(
    r"कितना|कितनी|कितने|किती|मात्रा|प्रमाण|खुराक|डोस|गणना|हिशोब|प्रति\s*(एकड़|एकर|हेक्टर|हेक्टेयर)|एकड़|एकर|हेक्टर|हेक्टेयर|किलो|बोरी|पोती")
_ANYDIGIT = re.compile(r"[\d०-९]")

REFUSAL = (
    "I'm sorry, I can't help with that. AgriAdvisor does **not** recommend pesticides, "
    "herbicides or fungicides, and does **not** calculate fertilizer quantities or doses.\n\n"
    "For chemical crop protection or fertilizer plans, please contact your local "
    "**Krishi Vigyan Kendra (KVK)**, agriculture extension officer or a certified agronomist "
    "(and get a soil test done first).\n\n"
    "I can still help with crop cultivation practices such as climate and soil needs, land "
    "preparation, sowing, irrigation, general crop management, non-chemical pest and disease "
    "awareness, harvesting and storage."
)


@traceable(name="safety_question_check", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def is_blocked_question(q: str) -> bool:
    """True if the user asks for chemical product advice or fertilizer maths."""
    if _CHEM_RE.search(q):
        # Asking *what a pest looks like* is fine; asking for chemicals is not.
        return True
    if _FERT_RE.search(q) and _CALC_RE.search(q):
        return True
    if _RECOMMEND_PEST_RE.search(q):
        return True
    if _DEV_CHEM.search(q):
        return True
    if _DEV_FERT.search(q) and _DEV_CALC.search(q):
        return True
    return False


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_FERT_NUMERIC = re.compile(
    rf"\b({FERTILIZER_TERMS})\b.*\d|\d.*\b({FERTILIZER_TERMS})\b|\d+\s*[:\-]\s*\d+\s*[:\-]\s*\d+", re.I)


@traceable(name="safety_answer_scrub", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def scrub_answer(answer: str):
    """Remove any sentence that mentions chemical products or fertilizer doses.
    Returns (clean_answer, was_modified)."""
    kept, modified = [], False
    for line in answer.split("\n"):
        parts = re.split(r"(?<=[.!?।])\s+", line)
        out = []
        for s in parts:
            if (_CHEM_RE.search(s) or _FERT_NUMERIC.search(s) or _DEV_CHEM.search(s)
                    or (_DEV_FERT.search(s) and _ANYDIGIT.search(s))):
                modified = True
            else:
                out.append(s)
        kept.append(" ".join(out))
    clean = "\n".join(kept)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if modified:
                clean += "\n\n_Chemical and fertilizer details are not provided. Please ask your local KVK._"
    return clean, modified
