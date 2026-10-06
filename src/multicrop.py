"""Multi-crop support: detect all crops mentioned and handle comparison questions."""
import re
from typing import List, Optional
from langsmith import traceable

from . import config
from . import config


@traceable(name="detect_crops", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def detect_all_crops(question: str) -> List[str]:
    """Return all crops mentioned in the question (English, Hindi or Marathi)."""
    t = re.sub(r"[_\-.]+", " ", question.lower())
    t = re.sub(r"\s+", " ", t)
    matches = []
    for crop, aliases in config.CROP_ALIASES.items():
        for a in aliases:
            a = re.sub(r"[_\-.]+", " ", a)
            if a.isascii():
                pattern = re.compile(rf"\b{re.escape(a)}s?\b")
            else:
                pattern = re.compile(re.escape(a))
            matches.extend((match.start(), crop) for match in pattern.finditer(t))

    detected = []
    for _, crop in sorted(matches):
        if crop not in detected:
            detected.append(crop)
    return detected


def is_comparison_question(question: str) -> bool:
    """True if question asks to compare crops (vs, vs., or, difference, better, best, etc.)."""
    return bool(re.search(
        r"\b(vs\.?|versus|or|difference|compare|better|best|which|most suitable|recommend|"
        r"which one|either|similar)\b|तुलना|फरक|बेहतर|कौन सा|कौनसी|कोणते|कोणता|फरक",
        question.lower()
    ))


def get_comparison_intent(crops: List[str]) -> Optional[str]:
    """Return a short intent description for multi-crop questions."""
    if len(crops) > 1:
        return f"compare_{','.join(crops)}"
    return None


@traceable(name="separate_crop_subqueries", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def separate_query_per_crop(question: str, crops: List[str]) -> dict:
    """For each crop, create a separate query with that crop name included.
    
    Returns: {crop: modified_query}
    """
    if len(crops) <= 1:
        return {crops[0]: question} if crops else {}
    
    # Remove crop names from question for cleaner per-crop queries
    base = question
    aliases = sorted(
        ((a, crop) for crop, values in config.CROP_ALIASES.items() for a in values),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for alias, _ in aliases:
        if alias.isascii():
            base = re.sub(rf"\b{re.escape(alias)}s?\b", " ", base, flags=re.I)
        else:
            base = base.replace(alias, " ")

    base = re.sub(r"\b(vs\.?|versus|or|and|difference|between|among|compare|better|best|which|"
                  r"which one|either|similar)\b", "", base, flags=re.I)
    base = re.sub(r"तुलना|फरक|बेहतर|कौन सा|कौनसी|कोणते|कोणता", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    
    if not base:
        base = "cultivation practices"
    
    return {crop: f"{base} for {crop}" for crop in crops}


@traceable(name="split_query_aspects", run_type="chain", project_name=config.LANGSMITH_PROJECT)
def split_query_parts(question: str) -> List[str]:
    """Split explicit multi-question inputs while keeping single questions intact."""
    parts = [
        part.strip(" \t\r\n?؟")
        for part in re.split(r"(?<=[?؟])\s+", question)
        if part.strip(" \t\r\n?؟")
    ]
    return parts or [question.strip()]
