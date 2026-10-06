"""End-to-end RAG pipeline: greeting -> guard -> retrieve -> generate -> scrub."""
import re
import logging
from typing import Dict, List, Optional

from langsmith import traceable

from . import config, llm, safety
from .multicrop import detect_all_crops, is_comparison_question, split_query_parts
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .scope import is_out_of_scope, is_price_question, out_of_scope_reply, price_question_reply

logger = logging.getLogger(__name__)

GREETING_REPLY = ("Hello! 👋 I'm AgriAdvisor 🌾. Ask me about growing rice, wheat, maize, cotton, "
                  "soybean or sugarcane: climate, soil, sowing, irrigation, harvesting and storage.")
THANKS_REPLY = "You're welcome! 😊 Ask me anything else about rice, wheat, maize, cotton, soybean or sugarcane."
BYE_REPLY = "Goodbye! 🌾 Come back anytime you have a crop question."
ABOUT_REPLY = ("I'm AgriAdvisor 🌾, a crop advisory assistant. I answer questions on rice, wheat, maize, "
               "cotton, soybean and sugarcane (climate, soil, sowing, spacing, irrigation, harvesting, "
               "storage) using agriculture documents. I don't give pesticide, herbicide, fungicide or "
               "fertilizer-dose advice. Please ask your local KVK for those.")
NOT_FOUND = ("This information is not available in my knowledge base. "
             "Please consult your local Krishi Vigyan Kendra (KVK).")
CLARIFY_CROP = ("Which crop are you asking about? Please specify rice, wheat, maize, cotton, "
                "soybean or sugarcane.")
OUT_OF_SCOPE = ("I can only answer questions about the cultivation of rice, wheat, maize, cotton, "
                "soybean and sugarcane.")
PRICE_REPLY = ("I don't have live market prices. Please check Agmarknet or your local mandi / APMC "
               "for today's rates.")
SUPPORTED_LANGUAGES = ("English", "Hindi", "Marathi")
LOCALIZED_REPLIES = {
    "Hindi": {
        GREETING_REPLY: "नमस्ते! 👋 मैं AgriAdvisor हूँ 🌾। चावल, गेहूँ, मक्का, कपास, सोयाबीन या गन्ने की खेती के बारे में पूछें।",
        THANKS_REPLY: "आपका स्वागत है! 😊 चावल, गेहूँ, मक्का, कपास, सोयाबीन या गन्ने के बारे में और पूछें।",
        BYE_REPLY: "अलविदा! 🌾 खेती से जुड़ा कोई सवाल हो तो फिर आएँ।",
        ABOUT_REPLY: "मैं AgriAdvisor हूँ 🌾। मैं कृषि दस्तावेज़ों के आधार पर चावल, गेहूँ, मक्का, कपास, सोयाबीन और गन्ने की खेती संबंधी जानकारी देता हूँ। कीटनाशक या उर्वरक की मात्रा संबंधी सलाह के लिए अपने स्थानीय KVK से संपर्क करें।",
        NOT_FOUND: "यह जानकारी मेरे ज्ञान आधार में उपलब्ध नहीं है। कृपया अपने स्थानीय कृषि विज्ञान केंद्र (KVK) से सलाह लें।",
        CLARIFY_CROP: "आप किस फसल के बारे में पूछ रहे हैं? कृपया चावल, गेहूँ, मक्का, कपास, सोयाबीन या गन्ने में से बताएं।",
        OUT_OF_SCOPE: "मैं केवल चावल, गेहूँ, मक्का, कपास, सोयाबीन और गन्ने की खेती से जुड़े सवालों के जवाब दे सकता हूँ।",
        PRICE_REPLY: "मेरे पास आज के बाज़ार भाव उपलब्ध नहीं हैं। कृपया Agmarknet या अपनी स्थानीय मंडी/APMC में जाँच करें।",
        safety.REFUSAL: "माफ़ कीजिए, मैं कीटनाशक, खरपतवारनाशक या फफूंदनाशक की सलाह और उर्वरक की मात्रा या खुराक नहीं बता सकता। इसके लिए अपने स्थानीय कृषि विज्ञान केंद्र (KVK) या कृषि अधिकारी से संपर्क करें।",
    },
    "Marathi": {
        GREETING_REPLY: "नमस्कार! 👋 मी AgriAdvisor आहे 🌾. तांदूळ, गहू, मका, कापूस, सोयाबीन किंवा ऊस लागवडीबद्दल विचारा.",
        THANKS_REPLY: "आपले स्वागत आहे! 😊 तांदूळ, गहू, मका, कापूस, सोयाबीन किंवा उसाबद्दल आणखी विचारा.",
        BYE_REPLY: "पुन्हा भेटू! 🌾 शेतीविषयक प्रश्न असल्यास कधीही विचारा.",
        ABOUT_REPLY: "मी AgriAdvisor आहे 🌾. कृषी दस्तऐवजांवर आधारित तांदूळ, गहू, मका, कापूस, सोयाबीन आणि ऊस लागवडीची माहिती देतो. कीटकनाशक किंवा खताच्या मात्रेबाबत सल्ल्यासाठी स्थानिक KVK शी संपर्क साधा.",
        NOT_FOUND: "ही माहिती माझ्या ज्ञानसंग्रहात उपलब्ध नाही. कृपया आपल्या स्थानिक कृषी विज्ञान केंद्राशी (KVK) संपर्क साधा.",
        CLARIFY_CROP: "आपण कोणत्या पिकाबद्दल विचारत आहात? कृपया तांदूळ, गहू, मका, कापूस, सोयाबीन किंवा ऊस यापैकी पीक सांगा.",
        OUT_OF_SCOPE: "मी फक्त तांदूळ, गहू, मका, कापूस, सोयाबीन आणि ऊस लागवडीशी संबंधित प्रश्नांची उत्तरे देऊ शकतो.",
        PRICE_REPLY: "माझ्याकडे आजचे बाजारभाव उपलब्ध नाहीत. कृपया Agmarknet किंवा आपल्या स्थानिक बाजार समिती/मंडईत तपासा.",
        safety.REFUSAL: "क्षमस्व, मी कीटकनाशक, तणनाशक किंवा बुरशीनाशकाची शिफारस तसेच खताचे प्रमाण किंवा मात्रा सांगू शकत नाही. यासाठी आपल्या स्थानिक KVK किंवा कृषी अधिकाऱ्यांशी संपर्क साधा.",
    },
}
LOCALIZED_SCRUB_NOTE = {
    "Hindi": "\n\n_रासायनिक और उर्वरक संबंधी जानकारी नहीं दी गई है। कृपया अपने स्थानीय KVK से पूछें।",
    "Marathi": "\n\n_रासायनिक आणि खतासंबंधी माहिती दिलेली नाही. कृपया आपल्या स्थानिक KVK शी संपर्क साधा._",
}

_GREET = re.compile(
    r"(नमस्ते|नमस्कार|हॅलो|हेलो|हाय|h+i+|h+e+l+l*o+|h+e+y+|hi+ there|yo+|hola|namaste|namaskar|sup|"
    r"good (morning|afternoon|evening|day)|how are you( doing)?|how r u|"
    r"what'?s ?up|wassup)( (there|everyone|all|sir|madam|bro|bhai|advisor|agriadvisor))?")
_GREET_INTRO = re.compile(
    r"^(?:hi+|hello+|hey+|namaste|namaskar|good (?:morning|afternoon|evening|day)) "
    r"(?:i am|i'm|this is|my name is) [\w'-]+(?: [\w'-]+){0,2}$",
    re.I,
)
_THANKS = re.compile(r"(धन्यवाद|आभारी आहे|शुक्रिया|thanks?|thank you|thx|ty|ok|okay|great|cool|nice|good|got it|alright)"
                     r"( (a lot|so much|very much|you|bro|bhai))*")
_BYE = re.compile(r"(अलविदा|पुन्हा भेटू|बाय|bye( bye)*|goodbye|good night|see you|see ya|take care|cya)( (now|all|bro|bhai|later|soon))?")
_ABOUT = re.compile(r"(who are you|what are you|what is your name|what'?s your name|what can you do|"
                    r"what do you do|help|help me|how do you work|what can i ask)")


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s']", " ", t)       # drop punctuation / emojis
    return re.sub(r"\s+", " ", t).strip()


def smalltalk_reply(text: str) -> Optional[str]:
    """Return a canned reply for greetings / thanks / goodbye / 'who are you', else None."""
    t = _normalize(text)
    if not t or len(t) > 40:
        return None
    if _BYE.fullmatch(t):
        return BYE_REPLY
    if _GREET.fullmatch(t) or _GREET_INTRO.fullmatch(t):
        return GREETING_REPLY
    if _THANKS.fullmatch(t):
        return THANKS_REPLY
    if _ABOUT.fullmatch(t):
        return ABOUT_REPLY
    return None


# ------------------------------------------------------------------ helpers
_DEV = re.compile(r"[\u0900-\u097F]")
_MR = re.compile(r"आहे|आहेत|कशी|कसे|कसा|करावी|करावे|करावा|कधी|काय|साठी|मध्ये|नाही|आणि|किती|पाहिजे|द्यावे|घ्यावी|ची\b|चे\b|चा\b")
_HI = re.compile(r"\bहै\b|\bहैं\b|कैसे|कैसी|\bकब\b|क्या|\bमें\b|करें|कितना|कितनी|नहीं|\bऔर\b|चाहिए|डालें|\bकी\b|\bका\b|\bके\b")


def response_language(question: str, selected: str, previous: Optional[str] = None,
                      force_selected: bool = False) -> str:
    """Honor the selected language, detect Devanagari input, and retain chat language."""
    if selected != "English" or force_selected:
        return selected
    if _DEV.search(question):
        return "Marathi" if len(_MR.findall(question)) > len(_HI.findall(question)) else "Hindi"
    return previous if previous in SUPPORTED_LANGUAGES else "English"


GLOSSARY = {
    "Marathi": ("Use standard Marathi farming terms: high-quality seed = उत्तम प्रतीचे बियाणे, "
                "seed quality = बियाण्याची गुणवत्ता, germination test = उगवण क्षमता चाचणी, "
                "floating seeds = तरंगणारी बियाणे, land preparation = जमिनीची पूर्वमशागत, "
                "transplanting = पुनर्लागवड, plough = नांगरणी, puddling = चिखलणी, "
                "nursery = रोपवाटिका, seedling = रोप, sowing = पेरणी, weeding = निंदणी/तण काढणी, weed = तण, "
                "harvest = कापणी/काढणी, irrigation = सिंचन, drainage = निचरा, flowering = फुलोरा, panicle = लोंबी, "
                "grain = दाणे, soil = माती, yield = उत्पादन, spacing = अंतर, seed rate = बियाणे प्रमाण, "
                "Kharif = खरीप, Rabi = रब्बी, maturity = परिपक्वता, storage = साठवण, days = दिवस, cm = सेमी."),
    "Hindi": ("Use standard Hindi farming terms: high-quality seed = उच्च गुणवत्ता वाला बीज, "
              "seed quality = बीज की गुणवत्ता, germination test = अंकुरण परीक्षण, "
              "floating seeds = तैरते हुए बीज, land preparation = भूमि की तैयारी, "
              "transplanting = रोपाई, plough = जुताई, puddling = कादो/पडलिंग, "
              "nursery = पौधशाला/नर्सरी, seedling = पौध, sowing = बुवाई, weeding = निराई-गुड़ाई, weed = खरपतवार, "
              "harvest = कटाई, irrigation = सिंचाई, drainage = जल निकासी, flowering = फूल आना, panicle = बाली, "
              "grain = दाना, soil = मिट्टी, yield = उपज, spacing = दूरी, seed rate = बीज दर, Kharif = खरीफ, "
              "Rabi = रबी, maturity = परिपक्वता, storage = भंडारण, days = दिन, cm = सेमी."),
}

_TRANSLATED_LABELS = {
    "Hindi": {
        "high-quality seed": "उच्च गुणवत्ता वाला बीज",
        "seed quality": "बीज की गुणवत्ता",
        "land preparation": "भूमि की तैयारी",
        "transplanting": "रोपाई",
        "sowing": "बुवाई",
        "irrigation": "सिंचाई",
        "harvesting": "कटाई",
        "harvest": "कटाई",
        "soil": "मिट्टी",
        "water management": "जल प्रबंधन",
    },
    "Marathi": {
        "high-quality seed": "उत्तम प्रतीचे बियाणे",
        "seed quality": "बियाण्याची गुणवत्ता",
        "land preparation": "जमिनीची पूर्वमशागत",
        "transplanting": "पुनर्लागवड",
        "sowing": "पेरणी",
        "irrigation": "सिंचन",
        "harvesting": "कापणी",
        "harvest": "कापणी",
        "soil": "माती",
        "water management": "पाणी व्यवस्थापन",
    },
}


def _translate_labels(text: str, language: str) -> str:
    """Translate bullet labels deterministically even if the model preserves English headings."""
    labels = _TRANSLATED_LABELS.get(language, {})
    return re.sub(
        r"(?m)^(\s*[-*•]\s*\*\*)([^*:\n]+)(:\*\*)",
        lambda match: (
            match.group(1)
            + labels.get(match.group(2).strip().lower(), match.group(2).strip())
            + match.group(3)
        ),
        text,
    )


def _history_text(history: Optional[List[Dict]]) -> str:
    out = []
    for m in (history or [])[-6:]:
        role = "Farmer" if m["role"] == "user" else "Advisor"
        out.append(f"{role}: {m['content'][:400]}")
    return "\n".join(out)


def _history_crops(history: Optional[List[Dict]]) -> List[str]:
    """Crops of the most recent user message that named any."""
    for message in reversed(history or []):
        if message.get("role") == "user":
            crops = detect_all_crops(message.get("content", ""))
            if crops:
                return crops
    return []


@traceable(name="rewrite_query", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def _rewrite_query(question: str, crops: List[str], history: Optional[List[Dict]]) -> str:
    """Standalone ENGLISH search query: resolves follow-ups ('it', 'that') and translates
    Hindi/Marathi questions (the embedding model and the PDFs are English)."""
    english = question.isascii()
    crop_txt = " and ".join(crops)
    in_question = detect_all_crops(question)
    if english and not history:
        return question if set(crops) <= set(in_question) else f"{question} for {crop_txt}"

    fallback = (question if set(crops) <= set(in_question) else f"{question} for {crop_txt}") \
        if english else f"{crop_txt} cultivation practices"
    try:
        rewritten = llm.chat([
            {"role": "system", "content": (
                "Rewrite the farmer's latest question as ONE concise, standalone English "
                "agriculture search query. Use the recent conversation to resolve references "
                "such as 'it', 'that', 'what factors'. Translate Hindi/Marathi into English. "
                f"The crop(s) under discussion: {crop_txt}. Never change them. "
                "Return only the query; do not answer the question.")},
            {"role": "user", "content": (
                f"Recent conversation:\n{_history_text(history) or '(none)'}\n\n"
                f"Crop(s): {crop_txt}\nLatest question: {question}")},
        ], temperature=0, max_tokens=800).strip().strip("\"'")  # reasoning models need room
    except Exception:
        logger.warning("Query rewrite failed; using the deterministic crop-aware fallback", exc_info=True)
        return fallback

    found = detect_all_crops(rewritten)
    if not rewritten or not set(found) <= set(crops):
        return fallback
    return rewritten if found else f"{rewritten} for {crop_txt}"


@traceable(name="retrieve_crop_evidence", run_type="retriever", project_name=config.LANGSMITH_PROJECT)
def _retrieve(retriever, query: str, crops: List[str], k: int):
    if len(crops) == 1:
        return retriever.search(query, k=k, crop=crops[0])
    per = max(3, k // len(crops) + 1)           # comparison: fetch passages for EACH crop
    found = []
    for c in crops:
        found += retriever.search(f"{query} {c}", k=per, crop=c)
    return found


_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUM = re.compile(r"\d+(?:\.\d+)?")


_MONTH_ALIASES = {
    "january": ("January", "जनवरी"),
    "february": ("February", "फरवरी", "फेब्रुवारी"),
    "march": ("March", "मार्च"),
    "april": ("April", "अप्रैल", "एप्रिल"),
    "may": ("May", "मई", "मे"),
    "june": ("June", "जून"),
    "july": ("July", "जुलाई", "जुलै"),
    "august": ("August", "अगस्त", "ऑगस्ट"),
    "september": ("September", "सितंबर", "सप्टेंबर"),
    "october": ("October", "अक्टूबर", "ऑक्टोबर"),
    "november": ("November", "नवंबर", "नोव्हेंबर"),
    "december": ("December", "दिसंबर", "डिसेंबर"),
}


def _numbers(text: str) -> set:
    """Numbers and (capitalised) month names: both must be backed by the passages."""
    t = text.translate(_DIGITS)
    months = set()
    for english, aliases in _MONTH_ALIASES.items():
        for alias in aliases:
            if alias.isascii():
                pattern = rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])"
            else:
                pattern = rf"(?<![\u0900-\u097F]){re.escape(alias)}(?![\u0900-\u097F])"
            if re.search(pattern, t, re.I):
                months.add(english)
                break
    return set(_NUM.findall(t)) | months


def _remove_tags_and_unsupported_numbers(text: str) -> str:
    """Remove non-citation bracket tags while preserving numeric source citations."""
    return re.sub(r"\[(?!\d+\])[^\]\n]{1,30}\]", "", text)


@traceable(name="validate_quantities", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def _ground_numbers(text: str, passages) -> tuple:
    """Drop any sentence containing a number that appears in none of the retrieved passages
    (guard against invented dates, spacings, rates). Returns (text, dropped_any)."""
    ctx_text = " ".join(p["text"] for p in passages)
    allowed = _numbers(ctx_text)
    kept_lines, dropped = [], False
    for line in text.split("\n"):
        out = []
        for sent in re.split(r"(?<=[.!?।])\s+", line):
            body = re.sub(r"\[\d+\]", "", sent)
            body = re.sub(r"^\s*(?:[-*•]\s*)?\d+[.)]\s+", "", body)  # list numbering
            if _numbers(body) - allowed:
                dropped = True
            else:
                out.append(sent)
        kept_lines.append(" ".join(out))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip(), dropped


def _loc(language: str, text: str) -> str:
    return LOCALIZED_REPLIES.get(language, {}).get(text, text)


NOTE_EN = "\n\n_Chemical and fertilizer details are not provided. Please ask your local KVK._"


@traceable(name="validate_answer_evidence", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def _verify(text: str, passages) -> tuple:
    """Second-pass fact check: drop bullets that the passages do not support (invented practices,
    dates, equipment). Returns (text, dropped_any). On any failure the text is kept unchanged."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return text, False
    ctx = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, 1))
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        cleaned = re.sub(r"\[\d+\]", "", line).strip()
        numbered_lines.append(f"{i}. {cleaned}")
    numbered = "\n".join(numbered_lines)
    try:
        verdict = llm.chat([
            {"role": "system", "content": (
                "You are a strict agricultural fact checker. The CONTEXT is the only source of truth. "
                "For each numbered statement answer YES only if the context explicitly states it (or a "
                "plain paraphrase of it). Answer NO if it adds anything the context does not say: an "
                "invented practice, date, equipment, field layout, or advice adapted to a condition the "
                "context does not mention. Output exactly one line per statement, like '3: NO'. Nothing else.")},
            {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nSTATEMENTS:\n{numbered}"},
        ], temperature=0, max_tokens=800)
    except Exception:
        logger.exception("Evidence validation failed for a retrieved answer")
        raise
    verdicts = {int(n): v.upper() for n, v in re.findall(r"(\d+)\s*[:.)-]\s*(YES|NO)", verdict, re.I)}
    if not verdicts:
        return text, False
    kept = [l for i, l in enumerate(lines, 1)
            if verdicts.get(i, "YES") == "YES" or "not covered" in l.lower()]
    return "\n".join(kept), len(kept) < len(lines)


@traceable(name="translate_verified_answer", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def _translate(text: str, language: str) -> Optional[str]:
    """Translate the verified answer and fail explicitly rather than returning English."""
    try:
        translated = llm.chat([
            {"role": "system", "content": (
                f"You are a professional agricultural translator. Translate the complete answer into natural, "
                f"farmer-friendly {language}. Translate each bold bullet label too: the word 'Label' in "
                "'- **Label:** text' is a format example, not a literal English label. Keep the bullet "
                "and bold-markup structure, but use a natural translated label. "
                "citation markers unchanged. Copy all numeric digits exactly; do not spell out or "
                "convert numbers. Keep every number and its meaning. Do not add, remove, "
                "merge or reorder facts. Translate every sentence and all farming terminology; do not "
                f"leave English prose in the answer. Return only the translation.\n{GLOSSARY.get(language, '')}")},
            {"role": "user", "content": text},
        ], temperature=0, max_tokens=2500).strip()
    except Exception as exc:
        raise RuntimeError(f"Could not translate the complete answer into {language}.") from exc
    if not translated or not _DEV.search(translated):
        raise RuntimeError(f"The model did not return a complete {language} answer.")
    if _numbers(re.sub(r"\[\d+\]", "", translated)) != _numbers(re.sub(r"\[\d+\]", "", text)):
        raise RuntimeError(f"The {language} translation did not preserve the verified quantities.")
    if set(re.findall(r"\[\d+\]", translated)) != set(re.findall(r"\[\d+\]", text)):
        raise RuntimeError(f"The {language} translation did not preserve the source citations.")
    return translated


# ------------------------------------------------------------------ main entry
@traceable(
    name="agriadvisor_rag",
    run_type="chain",
    project_name=config.LANGSMITH_PROJECT,
    process_inputs=lambda inputs: {key: value for key, value in inputs.items() if key != "retriever"},
)
def answer_question(question: str, retriever, history=None, crop: Optional[str] = None,
                    top_k: int = config.TOP_K, language: str = "English",
                    prev_language: Optional[str] = None,
                    force_language: bool = False) -> Dict:
    question = question.strip()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    language = response_language(question, language, prev_language, force_language)

    in_question = detect_all_crops(question)
    state = [c for c in (crop or "").split(",") if c in config.CROPS]
    crops = in_question or state or _history_crops(history)

    def reply(text, status, sources=None):
        return {"answer": text, "sources": sources or [], "status": status,
                "crop": ",".join(crops) if crops else None, "language": language}

    small = smalltalk_reply(question)
    if small:
        return reply(_loc(language, small), "smalltalk")
    if safety.is_blocked_question(question):
        return reply(_loc(language, safety.REFUSAL), "blocked")
    if is_price_question(question) and not re.search(r"cost of (cultivation|production)", question, re.I):
        return reply(price_question_reply(language), "price")
    if not in_question and is_out_of_scope(question):
        return reply(out_of_scope_reply(language), "out_of_scope")
    if not crops:
        return reply(_loc(language, CLARIFY_CROP), "needs_clarification")

    question_parts = split_query_parts(question)
    query_parts = (
        [_rewrite_query(part, crops, history) for part in question_parts]
        if len(question_parts) > 1 else [_rewrite_query(question, crops, history)]
    )
    standalone_query = " ".join(query_parts)
    if any(safety.is_blocked_question(part) for part in query_parts):
        return reply(_loc(language, safety.REFUSAL), "blocked")

    k = top_k + 3 if len(question.split()) > 18 else top_k   # long scenario questions need more context
    complex_query = (
        len(crops) > 1 or is_comparison_question(question)
        or len(question_parts) > 1 or len(question.split()) > 18
    )
    note = ""
    if complex_query:
        from .rag_agent import run_rag_agent

        agent_state = run_rag_agent(
            question, standalone_query, retriever, crops, _history_text(history), k,
            query_parts=query_parts,
        )
        passages = agent_state["passages"]
        note = agent_state["note"]
        text = agent_state["answer"]
    else:
        passages = _retrieve(retriever, query_parts[0], crops, k)
        asked = query_parts[0] if query_parts[0] == question else (
            f"{query_parts[0]}\n(Farmer's original wording: {question})")
        text = ""
    if not passages or max(p["score"] for p in passages) < config.MIN_SCORE:
        return reply(_loc(language, NOT_FOUND), "no_context")
    # drop weakly related passages (per crop): they invite off-topic or invented content
    best = {}
    for p in passages:
        best[p["crop"]] = max(best.get(p["crop"], 0.0), p["score"])
    passages = [p for p in passages if p["score"] >= max(config.MIN_SCORE * 0.5, 0.6 * best[p["crop"]])]

    if not complex_query:
        text = llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(asked, passages, _history_text(history), note)},
        ])
    text = _remove_tags_and_unsupported_numbers(text)
    text, modified = safety.scrub_answer(text)
    text = text.replace(NOTE_EN, "")
    text, ungrounded = _ground_numbers(text, passages)

    verified_dropped = False
    if len(question.split()) > 14 or len(crops) > 1 or len(question_parts) > 1:
        text, verified_dropped = _verify(text, passages)
    if len(re.sub(r"[\s\[\]\d*_\-•.,;:।]", "", text.replace("Not covered", ""))) < 15:
        return reply(_loc(language, NOT_FOUND), "no_context")

    if language != "English":
        translated = _translate(text, language)
        if translated:
            translated = _translate_labels(translated, language)
            translated, m2 = safety.scrub_answer(translated)
            translated = translated.replace(NOTE_EN, "")
            modified = modified or m2
            text = translated
    asked_inputs = re.search(r"pesticid|insectic|herbicid|fungicid|fertili|urea|npk|spray|manure|कीटनाशक|कीटकनाशक|खत|खाद|उर्वरक|फवारणी|छिड़काव",
                             question, re.I)
    if modified and asked_inputs:
        text += NOTE_EN if language == "English" else LOCALIZED_SCRUB_NOTE[language]

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(passages)})
    sources = [{"n": n, **passages[n - 1]} for n in (cited or range(1, len(passages) + 1))]
    status = "scrubbed" if modified else ("ungrounded_trimmed" if (ungrounded or verified_dropped) else "ok")
    return reply(text, status, sources)
