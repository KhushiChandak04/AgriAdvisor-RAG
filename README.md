# 🌾 AgriAdvisor: Agriculture Advisory RAG Chatbot

A Streamlit chatbot that answers crop-production questions for **rice, wheat, maize, cotton, soybean and sugarcane**
using only your agriculture PDFs, and shows **source citations** (file + page + excerpt).

**Stack:** PyMuPDF (+ Tesseract OCR for scanned pages) → heading/paragraph-aware chunks → sentence-transformers
`all-MiniLM-L6-v2` → FAISS + BM25 hybrid retrieval → LangGraph workflow for complex and multi-crop queries →
Groq → Streamlit. Optional LangSmith tracing captures pipeline stages, retrieved context, prompts, model latency and errors.

**Safety rules (enforced in code, not only in the prompt)**
- No pesticide / herbicide / fungicide recommendations or spray schedules.
- No fertilizer calculations, doses or ratios.
- Three layers: question filter → restrictive system prompt → answer scrubber that deletes any sentence mentioning chemicals or fertilizer quantities.
- Such questions get a polite refusal pointing to the local KVK / extension officer.

## Important: the PDFs are NOT included
The build environment had no access to download crop guides, so `data/` is empty and `vectorstore/` is empty (the old HR-policy
index does not exist in this package, and `python -m src.ingest` deletes any old index before rebuilding).
You must add agriculture PDFs (step 4) and build the index (step 5). Nothing is invented: with no PDFs the app will not run answers.

## Windows setup (Command Prompt or PowerShell, Python 3.10-3.12)

```bat
:: 1. Unzip, then open a terminal in the project folder
cd agri_advisor

:: 2. Create venv + install everything
setup_windows.bat
::    (manual equivalent)
::    python -m venv venv
::    venv\Scripts\activate
::    pip install -r requirements.txt

:: 3. Add your free Groq key: https://console.groq.com/keys
::    open .env (created from .env.example) and set GROQ_API_KEY=...
notepad .env
```

#### Enable LangSmith tracing (optional)
1. Sign in to [LangSmith](https://smith.langchain.com/) and create a project (for example, `agri_advisor`).
2. In **Settings → API keys**, create a key with tracing access. Copy it once; do not put it in source control or share it.
3. Open the project-root `.env` file and set:
   ```dotenv
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_new_langsmith_api_key
   LANGSMITH_PROJECT=agri_advisor
   ```
   Leave `LANGSMITH_ENDPOINT` unset for the default LangSmith Cloud region. If your workspace is hosted in another
   region or a self-hosted deployment, set it to the endpoint shown by that workspace.
4. Save `.env`, stop the running Streamlit process with `Ctrl+C`, and restart it with `run_app.bat`.
5. Ask a crop question in the app. Open the matching project in LangSmith and check **Tracing/Runs**; the pipeline
   creates child runs for crop detection, query splitting/rewriting, retrieval, reranking, evidence and quantity
   validation, Groq calls, translation, and safety checks.

To test LangSmith independently of the app, run this from the project folder after configuring `.env`:
```bat
venv\Scripts\python scripts\check_langsmith.py
```
The check validates access to the configured project, submits one synthetic run, flushes it, and reads it back.
It does not include a farmer question, prompt, or document content.

If no runs appear, confirm that you edited the project-root `.env` (not `.env.example`), that tracing is exactly
`true`, that the API key is valid, and that the app was restarted. Check firewall/proxy access to the LangSmith
endpoint. Traces can contain questions, conversation context, prompts, and retrieved document excerpts; only enable
tracing if that data may be sent to your LangSmith workspace.

If a real API key was ever placed in `.env.example` or committed to source control, revoke it in LangSmith and
create a replacement before enabling tracing.

### 4. Add the agriculture PDFs
Option A: automatic best effort
```bat
venv\Scripts\activate
python scripts\download_pdfs.py
```

Option B (most reliable): download PDFs in your browser into `data\`. Good free sources: FAO crop guides (fao.org, openknowledge.fao.org),
ICAR (krishi.icar.gov.in), state agriculture universities (MPKV Rahuri, VNMKV Parbhani, TNAU) package-of-practices documents.
**The crop name must be in the file name**, e.g. `rice_package_of_practices.pdf`, `sugarcane_icar.pdf`. Use only crop PDFs: other files are skipped.

### 5. Build the fresh vectorstore (also deletes any old one)
```bat
run_ingest.bat
:: or:  venv\Scripts\activate  &&  python -m src.ingest
```
First run downloads the embedding model (~90 MB). **Re-run this whenever you add or change PDFs.**
Each newly ingested chunk includes `crop`, `section`, `page`, and `source`. Rebuild the vectorstore
after upgrading this version so existing chunks gain section metadata.

**⚠️ Knowledge Base Quality:**
The ingestion process will show per-crop chunk counts. For best accuracy, each crop should have at least 150 chunks.
If a crop has <150 chunks, add more PDFs to improve coverage and answer quality.

### 6. Start the chatbot
```bat
run_app.bat
:: or:  venv\Scripts\activate  &&  streamlit run app.py
```
Open http://localhost:8501.

## OCR for scanned PDFs (optional)
Pages with almost no text are OCR'd automatically if Tesseract is installed:
1. Install from https://github.com/UB-Mannheim/tesseract/wiki (default path `C:\Program Files\Tesseract-OCR`).
2. Either add that folder to PATH, or set in `.env`:
   `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`
3. Re-run `run_ingest.bat`. Sources from OCR are marked "(OCR)" in the UI.

## Project layout
```
agri_advisor/
├── app.py                  Streamlit chat UI + source citations
├── requirements.txt  .env.example  setup_windows.bat  run_ingest.bat  run_app.bat
├── data/                   put crop PDFs here
├── vectorstore/            built by ingest (FAISS index + chunks + meta)
├── scripts/download_pdfs.py, pdf_sources.json
├── src/
│   ├── config.py           settings, crop list/aliases
│   ├── pdf_processing.py   extraction, OCR fallback, heading/paragraph-aware chunking
│   ├── ingest.py           wipes old index, builds fresh one
│   ├── retriever.py        FAISS + BM25 hybrid (RRF), crop filter
│   ├── rag_agent.py        LangGraph detect → sub-query → retrieve → rerank → validate → answer
│   ├── prompts.py          agriculture RAG system prompt
│   ├── safety.py           chemical/fertilizer guardrails
│   ├── suggestions.py      localized starter questions
│   ├── llm.py              Groq client
│   └── rag_pipeline.py     guard → retrieve → generate → scrub → cite
└── tests/test_safety.py    python tests\test_safety.py
```

Each chat keeps its own recent messages, active crop(s), and answer language in Streamlit session state.
Follow-up questions are rewritten into standalone, crop-specific retrieval queries. Comparison queries
retrieve each crop separately, rerank and validate the evidence, and retain crop, section, page and source
metadata through answer generation and citations. If no crop has been established, AgriAdvisor asks the user
to specify one before searching the knowledge base. Use the **Language** menu to choose English, Hindi or Marathi;
Devanagari input is also detected and the language is remembered per chat.

## Troubleshooting
- **"Knowledge base not ready"**: you haven't run `run_ingest.bat` after adding PDFs.
- **"GROQ_API_KEY is missing"**: edit `.env`, restart the app.
- **"no usable PDFs"**: file names must contain a crop name.
- **Answers say "not available in my knowledge base"**: the PDFs don't cover that question, or lower `MIN_SCORE` in `.env`.
- **Slow first start**: the embedding model is being downloaded/cached.
- If `faiss-cpu` fails to install, use Python 3.11 or 3.12 (64-bit).

Answers are generated from the supplied documents and may be incomplete. Confirm important decisions with your local Krishi Vigyan Kendra.
