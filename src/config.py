"""Central configuration for AgriAdvisor."""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
VECTORSTORE_DIR = ROOT / "vectorstore"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes"}
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "agriadvisor")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))

CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 150    # characters
OCR_MIN_CHARS = 50     # a page with less text than this is OCR'd

# Only these crops are allowed in the knowledge base.
CROP_ALIASES = {
    "rice": ["rice", "paddy", "चावल", "धान", "तांदूळ", "तांदुळ", "भात"],
    "wheat": ["wheat", "गेहू", "गेहूं", "गहू", "गव्ह"],
    "maize": ["maize", "corn", "मक्का", "मकई", "मका", "मक्य"],
    "cotton": ["cotton", "कपास", "कापूस", "कापस"],
    "soybean": ["soybean", "soyabean", "soya", "सोयाबीन", "सोयाबिन"],
    "sugarcane": ["sugarcane", "sugar cane", "sugar-cane", "sugar_cane", "गन्ना", "गन्ने", "ईख", "ऊस", "उसा"],
}
CROPS = list(CROP_ALIASES)


def detect_crop(text: str):
    """Return the crop key found in a filename/question (English, Hindi or Marathi), else None."""
    t = re.sub(r"[_\-.]+", " ", text.lower())
    t = re.sub(r"\s+", " ", t)
    for crop, aliases in CROP_ALIASES.items():
        for a in aliases:
            a = re.sub(r"[_\-.]+", " ", a)
            if a.isascii():
                if re.search(rf"\b{re.escape(a)}s?\b", t):
                    return crop
            elif re.search(rf"(?<![\u0900-\u097F]){re.escape(a)}", t):  # Devanagari: allow suffixes
                return crop
    return None
