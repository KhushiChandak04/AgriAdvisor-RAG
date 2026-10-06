"""Keep unit tests deterministic and prevent accidental external LLM requests."""
import re

import pytest


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    def fake_chat(messages, **kwargs):
        system = messages[0]["content"].lower()
        user = messages[-1]["content"]
        if "translator" in system:
            answer = user.replace("Sowing", "पेरणी").replace("Rice", "तांदूळ")
            answer = answer.replace("June", "जून").replace("Harvest", "कापणी")
            return answer.replace(":", "ी:")
        if "fact checker" in system:
            statement_ids = re.findall(r"(?m)^(\d+)\.\s", user)
            return "\n".join(f"{number}: YES" for number in statement_ids)
        if "rewrite the farmer" in system:
            crops = re.findall(r"rice|wheat|maize|cotton|soybean|sugarcane", system)
            return f"cultivation practices for {crops[-1]}" if crops else "crop cultivation practices"
        if "comparison of:" in user.lower():
            return "- **Sowing:** Rice: season [1]; Wheat: season [2]."
        return "- **Evidence:** The crop document contains relevant guidance [1]."

    monkeypatch.setattr("src.rag_pipeline.llm.chat", fake_chat)
