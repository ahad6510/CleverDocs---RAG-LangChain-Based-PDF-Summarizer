

# ⚡ CleverDocs: Intelligent RAG & AI-Powered Document Analysis

**CleverDocs** is a production-ready, dark-mode **RAG (Retrieval-Augmented Generation)** application that transforms how you interact with your PDFs. Moving beyond simple text extraction, CleverDocs features secure authentication, global rate limiting, and an advanced AI-OCR fallback system to read even the most complex, messy, or handwritten documents (like tickets and receipts) with surgical precision. 🧠💨

🚀 LIVE APP: https://cleverdocs---rag-langchain-based-pdf-summarizer-cdetlxutxrsnzq.streamlit.app/

---

## ✨ Key Capabilities

* **🔐 Secure Auth:** Frictionless Google OAuth2 login with persistent browser cookies—no more "re-login" fatigue!
* **👁️ AI OCR Fallback:** Stuck with an unreadable/scanned PDF? CleverDocs auto-routes them through Google's elite Vision AI (`gemini-flash-latest`) for flawless extraction.
* **📊 Smart Quota Engine:** A custom-built, JSON-based global database tracks shared API usage with a slick sidebar progress bar. It even handles automatic midnight resets!
* **🔍 Vector Search Engine:** Built on **FAISS** and **HuggingFace** embeddings for lightning-fast, high-accuracy context retrieval.
* **💾 Context-Aware Chat:** Remembers your flow and allows you to export your entire knowledge session as a `.txt` file.
* **🎨 Modern UI:** A premium, Glassmorphism-inspired dark-mode interface.

---

## 🏗️ The Engineering Blueprint

### ⚙️ Under the Hood

| Category | Technology |
| --- | --- |
| **Orchestration** | `LangChain`, `Streamlit` |
| **Intelligence** | `Google Gemini 2.0 Flash` |
| **Vision/OCR** | `Google Gemini Flash Latest` |
| **Search/Retrieval** | `FAISS`, `all-MiniLM-L6-v2` |
| **Security** | `OAuth2`, `JWT`, `CookieController` |
| **Storage** | `Lazy-Evaluation` JSON Data-store |

---

## 🎓 Why CleverDocs?

Most RAG apps break the moment they hit a scanned PDF or a refresh button. CleverDocs is built for **resilience**:

1. **Error-Resilient:** Automatically handles missing cookies and expired OAuth states.
2. **Crash-Proof:** Gracefully locks chat inputs when global limits are hit, preventing `429` server crashes.
3. **Optimized:** Uses intelligent text chunking and overlap to ensure the AI never misses the "big picture."

---

## 👨‍💻 Developed By : Ahad khan with ❤️.
