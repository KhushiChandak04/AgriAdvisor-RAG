import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config, llm

print("Model:", config.GROQ_MODEL)
print("1) DNS lookup for api.groq.com ...")
try:
    print("   OK ->", socket.gethostbyname("api.groq.com"))
except Exception as e:
    print("   FAILED:", e)

print("2) Test call to Groq ...")
try:
    print("   OK ->", llm.chat([{"role": "user", "content": "Reply with the word OK"}], max_tokens=300))
except Exception as e:
    print("   FAILED:", e)