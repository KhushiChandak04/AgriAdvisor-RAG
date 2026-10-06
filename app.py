"""AgriAdvisor - Streamlit chatbot.  Run:  streamlit run app.py"""
import json
import re
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from src.i18n import LANG_NAMES, tr
from src.rag_pipeline import SUPPORTED_LANGUAGES, answer_question
from src.suggestions import suggest

st.set_page_config(page_title="AgriAdvisor", page_icon="🌾", layout="centered")

SPEECH_LANG = {"English": "en-IN", "Hindi": "hi-IN", "Marathi": "mr-IN"}

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 820px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#1B5E20 0%,#2E7D32 100%);}
[data-testid="stSidebar"] * {color: #F1F8E9 !important;}
[data-testid="stSidebar"] .stButton > button {background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.35);}
[data-testid="stSidebar"] .stButton > button:hover {background: rgba(255,255,255,.25);}
.agri-hero {background: linear-gradient(120deg,#2E7D32 0%,#66BB6A 60%,#C5E1A5 100%); color: #fff;
  border-radius: 18px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 4px 14px rgba(46,125,50,.25);}
.agri-hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
.agri-hero p {margin: 4px 0 0 0; opacity: .95; font-size: .95rem;}
[data-testid="stChatMessage"] {background: #FFFFFF; border: 1px solid #DCEDC8; border-left: 5px solid #66BB6A;
  border-radius: 14px; padding: 10px 14px; box-shadow: 0 1px 4px rgba(0,0,0,.05);}
.main .stButton > button {border-radius: 20px; border: 1.5px solid #66BB6A; background: #F1F8E9; color: #1B5E20;
  text-align: left; transition: all .15s;}
.main .stButton > button:hover {background: #66BB6A; color: #fff; border-color: #2E7D32;}
.agri-sugg {color:#2E7D32; font-weight:600; margin: 8px 0 4px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_retriever():
    from src.retriever import Retriever
    return Retriever()


def strip_citations(text: str) -> str:
    """Remove [1], [2][3] style markers from the answer."""
    return re.sub(r"(\s*\[\d+\])+", "", text)


def speech_text(text: str) -> str:
    t = re.sub(r"\*\*|__|`|#", "", strip_citations(text))
    t = re.sub(r"(?m)^\s*[-•]\s*", "", t)
    return re.sub(r"\s+", " ", t.replace("\n", ". ")).strip()


def speak_button(text: str, lang: str, label: str, stop: str, no_voice: str):
    """🔊 Text-to-speech through the browser's built-in Web Speech API (no extra packages).
    The button lives inside the component so the click counts as the user gesture browsers require."""
    payload = json.dumps(speech_text(text)).replace("</", "<\\/")
    code = SPEECH_LANG.get(lang, "en-IN")
    st.iframe(f"""
<button id="b" style="border:1.5px solid #66BB6A;background:#F1F8E9;color:#1B5E20;border-radius:18px;
 padding:5px 14px;cursor:pointer;font-size:14px">{label}</button>
<span id="m" style="color:#B71C1C;font-size:12px;margin-left:8px"></span>
<script>
const text = {payload}, lang = "{code}", L = {json.dumps(label)}, S = {json.dumps(stop)};
const b = document.getElementById('b'), m = document.getElementById('m');
function voice() {{
  const v = speechSynthesis.getVoices(), l = lang.toLowerCase();
  return v.find(x => x.lang.toLowerCase().replace('_','-') === l) ||
         v.find(x => x.lang.toLowerCase().startsWith(l.slice(0,2))) ||
         (l.startsWith('mr') ? v.find(x => x.lang.toLowerCase().startsWith('hi')) : null);  // Marathi -> Hindi voice
}}
b.onclick = () => {{
  if (!('speechSynthesis' in window)) {{ m.textContent = {json.dumps(no_voice)}; return; }}
  if (speechSynthesis.speaking) {{ speechSynthesis.cancel(); b.textContent = L; return; }}
  const u = new SpeechSynthesisUtterance(text), v = voice();
  u.lang = lang; u.rate = 0.95;
  if (v) {{ u.voice = v; u.lang = v.lang; m.textContent = ''; }}
  else if (!lang.startsWith('en')) {{ m.textContent = {json.dumps(no_voice)}; }}
  u.onend = u.onerror = () => {{ b.textContent = L; }};
  b.textContent = S; speechSynthesis.speak(u);
}};
speechSynthesis.onvoiceschanged = () => {{}};
</script>""", height=46, alt="Listen to the assistant response")


def new_chat():
    cid = uuid.uuid4().hex[:8]
    st.session_state.chats[cid] = {"title": None, "messages": [], "crop": None, "lang": None}
    st.session_state.current = cid


def delete_chat(cid):
    st.session_state.chats.pop(cid, None)
    if not st.session_state.chats:
        new_chat()
    elif st.session_state.current not in st.session_state.chats:
        st.session_state.current = list(st.session_state.chats)[-1]


def ask(q: str):
    st.session_state.pending = q          # set by a suggestion button, handled below


def mark_language_selected():
    st.session_state.answer_language_selected = True


if "answer_language" not in st.session_state:
    st.session_state.answer_language = "English"
if "answer_language_selected" not in st.session_state:
    st.session_state.answer_language_selected = False
L = st.session_state.answer_language

if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_chat()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🌾 AgriAdvisor")
    if st.button(tr(L, "new_chat"), use_container_width=True):
        new_chat()
        st.rerun()

    st.markdown(tr(L, "your_chats"))
    for cid in reversed(list(st.session_state.chats)):
        chat = st.session_state.chats[cid]
        c1, c2 = st.columns([5, 1])
        label = ("● " if cid == st.session_state.current else "") + (chat["title"] or tr(L, "new_chat_title"))
        if c1.button(label, key=f"open_{cid}", use_container_width=True):
            st.session_state.current = cid
            st.rerun()
        if c2.button("🗑️", key=f"del_{cid}", help=tr(L, "delete_chat")):
            delete_chat(cid)
            st.rerun()

    st.divider()
    if st.button(tr(L, "delete_all"), use_container_width=True):
        st.session_state.chats = {}
        new_chat()
        st.rerun()

    st.caption(tr(L, "safety_note"))

# ---------------- Header ----------------
title_column, language_column = st.columns([4, 2])
with title_column:
    st.markdown(f'<div class="agri-hero"><h1>🌾 AgriAdvisor</h1><p>{tr(L, "tagline")} · {tr(L, "caption")}</p></div>',
                unsafe_allow_html=True)
with language_column:
    st.selectbox(
        "🌐 Language / भाषा",
        SUPPORTED_LANGUAGES,
        key="answer_language",
        on_change=mark_language_selected,
        format_func=LANG_NAMES.get,
    )

try:
    with st.spinner(tr(L, "loading")):
        retriever = load_retriever()
except Exception as e:
    st.error(tr(L, "kb_error", err=e))
    st.info(tr(L, "kb_hint"))
    st.stop()

chat = st.session_state.chats[st.session_state.current]
messages = chat["messages"]
cid = st.session_state.current

# ---------------- Conversation ----------------
for i, msg in enumerate(messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(tr(L, "sources")):
                for source in msg["sources"]:
                    st.caption(
                        f"[{source['n']}] {source['source']} · {tr(L, 'page')} {source['page']} "
                        f"· {source['crop']} · {source.get('section', 'General')}"
                    )
                    st.write(source["text"])
        if msg["role"] == "assistant" and msg.get("speakable", True):
            speak_button(msg["content"], msg.get("lang", L), tr(L, "listen"), tr(L, "stop"), tr(L, "no_voice"))

# ---------------- Clickable suggested questions ----------------
last = messages[-1] if messages else None
chips = last.get("suggestions", []) if last and last["role"] == "assistant" else (suggest([], L) if not messages else [])
if chips:
    st.markdown(f'<div class="agri-sugg">{tr(L, "suggested")}</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for j, q in enumerate(chips[:4]):
        cols[j % 2].button(q, key=f"sg_{cid}_{len(messages)}_{j}", on_click=ask, args=(q,), use_container_width=True)

# ---------------- Input ----------------
typed = st.chat_input(tr(L, "placeholder"))
question = typed or st.session_state.pop("pending", None)

if question:
    history = list(messages)                                   # conversation memory for follow-ups
    if not messages:
        chat["title"] = question[:28] + ("..." if len(question) > 28 else "")
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner(tr(L, "spinner")):
            try:
                res = answer_question(question, retriever, history, crop=chat.get("crop"),
                                      language=L, prev_language=chat.get("lang"),
                                      force_language=st.session_state.answer_language_selected)
                chat["crop"] = res["crop"] or chat.get("crop")  # remember crop(s), e.g. "rice,wheat"
                chat["lang"] = res["language"]                  # remember answer language
                answer = res["answer"]
                extra = {"lang": res["language"], "suggestions": res.get("suggestions", []),
                         "sources": res.get("sources", [])}
            except Exception as e:
                answer, extra = f"⚠️ {e}", {"speakable": False, "suggestions": []}
        st.markdown(answer)
    messages.append({"role": "assistant", "content": answer, **extra})
    st.rerun()