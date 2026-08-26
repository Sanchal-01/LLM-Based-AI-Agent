<div align="center">

# 🔮 ClarificAI
### A Modular Retrieval-Augmented Generation (RAG) Agent for Grounded PDF Question Answering

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-00A98F?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge" />
</p>

**Upload any PDF → Ask anything → Get answers grounded *only* in that document.**
No hallucinations. No guessing. Just retrieval-backed, verifiable answers.

</p>
</div>

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Module Breakdown](#-module-breakdown)
5. [How the RAG Pipeline Works](#-how-the-rag-pipeline-works)
6. [Anti-Hallucination Guardrails](#-anti-hallucination-guardrails)
7. [Tech Stack](#-tech-stack)
8. [Project Structure](#-project-structure)
9. [Installation & Setup](#-installation--setup)
10. [Configuration](#-configuration)
11. [Running the App](#-running-the-app)
12. [Usage Walkthrough](#-usage-walkthrough)
13. [Tuning Retrieval Behaviour](#-tuning-retrieval-behaviour)
14. [Troubleshooting](#-troubleshooting)
15. [Roadmap](#-roadmap)
16. [Contributing](#-contributing)
17. [License](#-license)

---

## 🔎 Overview

**ClarificAI** is an end-to-end Retrieval-Augmented Generation (RAG) application that lets a user upload a PDF document and interrogate it in natural language. Instead of relying on an LLM's internal (and often outdated or hallucinated) knowledge, the system **retrieves the most relevant passages from the document itself** and passes them to the language model as grounded context — guaranteeing that every answer is traceable back to the source text.

The project is intentionally built as a set of **decoupled, single-responsibility modules** rather than one monolithic script. Each stage of the RAG pipeline (extraction → chunking → embedding → indexing → retrieval → generation) lives in its own file, so any component — the embedding model, the vector store, or the LLM — can be swapped out without touching the rest of the system.

The frontend is a custom-themed **Streamlit** chat interface with left/right message bubbles, an animated "thinking" indicator, and an inspectable context panel that shows exactly which chunks the model used to answer each question.

---

## ✨ Key Features

| Feature | Description |
|---|---|
|  **Zero-Hallucination Architecture** | A strict system prompt and a distance-threshold cutoff ensure the LLM only answers from retrieved context — otherwise it explicitly says the answer isn't in the document. |
|  **Modular Pipeline Design** | Each RAG stage is an independent Python module (`A` → `F`), making the system easy to extend, test, and debug in isolation. |
|  **Local Persistence & Caching** | Extracted text, chunks, embeddings, and the FAISS index are all cached to disk — re-uploading the same PDF skips redundant computation. |
|  **Transparent Retrieval** | Every answer comes with an expandable panel showing the exact chunks retrieved and their L2 distance scores, so results are fully auditable. |
|  **Polished Chat UI** | Custom-styled Streamlit interface with user/AI message bubbles, an animated typing indicator, and Enter-to-submit input. |
|  **Tunable Retrieval** | Sidebar controls let you adjust the similarity threshold and the number of retrieved chunks (top-k) without touching code. |
|  **Automatic Re-indexing** | Uploading a new PDF automatically clears stale caches and rebuilds the pipeline for the new document. |

---

## 🏗 System Architecture

```
                         ┌───────────────────────┐
                         │     User Uploads PDF  │
                         └───────────┬───────────┘
                                     ▼
                    ┌────────────────────────────────┐
                    │       A_Pdf_utils_01_.py       │
                    │  Extracts raw text from PDF    │
                    │  → extracted_text.txt          │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │       B_Chunking_02.py         │
                    │  Splits text into overlapping  │
                    │  passages → Chunks.pkl         │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │      C_Embeddings_03.py        │
                    │  Encodes chunks into dense     │
                    │  vectors (all-MiniLM-L6-v2)    │
                    │  → Embeddings.npy              │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │      D_Vector_Store_04.py      │
                    │  Builds a FAISS IndexFlatL2    │
                    │  → faiss_index.pkl             │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │     E_Semantic_Search_05.py    │
                    │  Embeds the user query, finds  │
                    │  top-k nearest chunks          │
                    │  → context chunks + distances  │
                    └───────────────┬────────────────┘
                                    ▼
                    ┌────────────────────────────────┐
                    │  F_Gemini_2.5_Flash.py / app.py│
                    │  Injects context into a strict │
                    │  prompt → Gemini 2.5 Flash     │
                    │  → grounded answer in chat UI  │
                    └────────────────────────────────┘
```

---

## 📦 Module Breakdown

| Module | Responsibility | Input | Output |
|---|---|---|---|
| **`A_Pdf_utils_01_.py`** | Text extraction from the uploaded PDF | `*.pdf` | `extracted_text.txt` |
| **`B_Chunking_02.py`** | Splits extracted text into overlapping chunks to preserve context across boundaries | `extracted_text.txt` | `Chunks.pkl`, `chunks_texts.pkl` |
| **`C_Embeddings_03.py`** | Converts text chunks into 384-dimensional dense vectors using `all-MiniLM-L6-v2` | `Chunks.pkl` | `Embeddings.npy` |
| **`D_Vector_Store_04.py`** | Builds and serializes a FAISS `IndexFlatL2` for nearest-neighbour search | `Embeddings.npy` | `faiss_index.pkl` |
| **`E_Semantic_Search_05.py`** | Embeds the user's query and retrieves the top-k most similar chunks with their distance scores | Query string, `faiss_index.pkl` | `[chunks]`, `[distances]` |
| **`F_Gemini_2.5_Flash.py`** | Sends retrieved context + query to Gemini under a strict, hallucination-resistant system prompt | Prompt, context chunks | Generated answer |
| **`app.py`** | Streamlit orchestration layer — UI, session state, caching triggers, and pipeline coordination | User interaction | Rendered web interface |

---

## 🔬 How the RAG Pipeline Works

1. **Extraction** — The uploaded PDF's raw text is pulled out and saved as a flat text file.
2. **Chunking** — The text is split into overlapping segments so that context isn't lost at chunk boundaries (a sentence split across two chunks still has some shared context).
3. **Embedding** — Each chunk is converted into a dense vector representation using a Sentence Transformer model, capturing its semantic meaning rather than just keywords.
4. **Indexing** — All chunk embeddings are loaded into a FAISS index, enabling fast approximate/exact nearest-neighbour search over potentially thousands of chunks.
5. **Retrieval** — When a user asks a question, the query is embedded the same way, and FAISS returns the *k* most semantically similar chunks along with their L2 distances.
6. **Generation** — The retrieved chunks are inserted into a tightly-scoped prompt sent to Gemini 2.5 Flash, which is instructed to answer *only* from that context.

---

## 🛡 Anti-Hallucination Guardrails

### 1. Distance-Threshold Cutoff

Before any generation happens, the system checks the L2 distance of the closest retrieved chunk:

$$\text{Distance}(q, c) = \sum_{i=1}^{n} (q_i - c_i)^2$$

If this distance exceeds a configurable **threshold** (default `1.5`), the pipeline **skips generation entirely** and returns:

> *"This information is not available in the PDF."*

This prevents the model from being handed irrelevant context and trying to "make something up" anyway.

### 2. Constrained System Prompt

Even when relevant context is found, the LLM is wrapped in an explicit instruction template:

```text
<|system|>
You are a strict PDF assistant. You ONLY answer from the context below.
RULES:
1. Use ONLY the context provided. No outside knowledge allowed.
2. If the question cannot be answered from the context, respond exactly:
   "This information is not available in the PDF."
3. Do NOT guess. Do NOT use training knowledge. Do NOT make up facts.
```

Together, these two layers — **retrieval-side filtering** and **generation-side constraints** — form a defense-in-depth approach against hallucinated answers.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend / UI** | Streamlit (custom CSS-themed chat interface) |
| **Vector Store** | FAISS (`faiss-cpu`) — `IndexFlatL2` |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` (PyTorch backend) |
| **Generative Model** | Google Gemini 2.5 Flash via `google-generativeai` |
| **PDF Parsing** | `pdfplumber` |
| **Data Handling** | `numpy`, `pickle` |
| **Language** | Python 3.10+ |

---

## 📁 Project Structure

```
LLM-Based-AI-Agent/
│
├── app.py                     # Streamlit app — UI & orchestration
├── A_Pdf_utils_01_.py          # PDF text extraction
├── B_Chunking_02.py            # Text chunking logic
├── C_Embeddings_03.py          # Embedding generation
├── D_Vector_Store_04.py        # FAISS index build/load
├── E_Semantic_Search_05.py     # Query embedding + retrieval
├── F_Gemini_2.5_Flash.py       # Gemini prompt construction & call
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes cache & credentials
│
└── (generated at runtime — not committed)
    ├── extracted_text.txt
    ├── Chunks.pkl
    ├── chunks_texts.pkl
    ├── Embeddings.npy
    └── faiss_index.pkl
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sanchal-01/LLM-Based-AI-Agent.git
cd LLM-Based-AI-Agent
```

### 2. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🔑 Configuration

The app requires a **Google Gemini API key**. Get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

**Option A — Environment variable (recommended):**

```bash
export GEMINI_API_KEY="your_api_key_here"      # macOS/Linux
setx GEMINI_API_KEY "your_api_key_here"        # Windows
```

**Option B — Directly in `app.py`:**

```python
GEMINI_API_KEY = "your_api_key_here"
```

> ⚠️ If you use Option B, make sure `app.py` is **never** committed to a public repository with your real key inside it.

---

## 🚀 Running the App

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧭 Usage Walkthrough

1. **Upload a PDF** from the sidebar file uploader.
2. Wait for the status panel to confirm the document has been **indexed** (chunked, embedded, and loaded into FAISS).
3. Type a question in the input box and press **Enter** (or click **Ask**).
4. Watch the animated "thinking" indicator while the query is embedded, matched against the index, and passed to Gemini.
5. Read the grounded answer — click **"Context used"** below it to inspect exactly which chunks and distance scores produced that answer.
6. Use **"Clear cache & chat"** in the sidebar to wipe the current index and conversation and start fresh with a new document.

---

## 🎛 Tuning Retrieval Behaviour

Two sidebar sliders let you control retrieval quality without editing code:

| Control | Effect |
|---|---|
| **Distance threshold** | Lower = stricter matching (more likely to say "not found"). Higher = more lenient (may retrieve loosely related chunks). Default: `1.5`. |
| **Context chunks (top-k)** | Number of chunks sent to Gemini as context. Higher values give the model more surrounding information but increase prompt size and (for narrow questions) may dilute relevance. Default: `2`. |

**Tip:** If answers seem incomplete for multi-part questions (e.g. "list the three types of X"), try increasing **top-k** to 3–5, and check that your chunking strategy (`B_Chunking_02.py`) doesn't split closely related content across too many chunks.

---

## 🧯 Troubleshooting

| Issue | Likely Cause | Fix |
|---|---|---|
| `"This information is not available in the PDF"` for something that *is* in the PDF | Distance threshold too strict, or relevant content split across chunks not in the top-k | Increase threshold slightly and/or increase top-k; review chunk size/overlap in `B_Chunking_02.py` |
| Answers get cut off mid-sentence | `max_output_tokens` too low in the Gemini generation config | Increase `max_output_tokens` in `F_Gemini_2.5_Flash.py` |
| Re-uploading the same PDF doesn't reflect edits | Cached artifacts from a previous run are being reused | Use **"Clear cache & chat"** in the sidebar, or manually delete the generated `.pkl` / `.npy` / `.txt` files |
| `ResourceExhausted` error from Gemini | API rate limit reached | Wait 20–30 seconds and retry; consider adding request throttling for production use |
| Slow first response after upload | Embedding model and/or FAISS index are being built for the first time | This is expected — subsequent queries on the same document reuse cached artifacts |

---

## 🗺 Roadmap

- [ ] Support for multi-PDF sessions (query across several documents at once)
- [ ] Swap `IndexFlatL2` for an approximate index (e.g. `IndexIVFFlat`) for large document sets
- [ ] Add citation highlighting — jump directly to the source page/paragraph in the PDF
- [ ] Streaming token-by-token responses instead of a single blocking call
- [ ] Dockerfile for one-command containerized deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Built with Python, FAISS, and Gemini 2.5 Flash
</div>
