

# ⚡ CleverDocs: Intelligent RAG & AI-Powered Document Analysis

**CleverDocs** is a production-ready, dark-mode **RAG (Retrieval-Augmented Generation)** application that transforms how you interact with your PDFs. Built for extreme speed and resilience, CleverDocs features secure authentication, global rate limiting, and a blazing-fast hybrid local OCR system to read even the most complex or scanned documents without relying on rate-limited cloud vision APIs. 🧠💨

**🚀 [Launch CleverDocs Live**](https://cleverdocs---rag-langchain-based-pdf-summarizer-cdetlxutxrsnzq.streamlit.app/)

---

## ✨ Key Capabilities

* **🔐 Secure Auth:** Frictionless Google OAuth2 login with persistent browser cookies—no more "re-login" fatigue!
* **⚡ Ultra-Fast Inference:** Powered by the **Groq API**, delivering lightning-fast conversational responses with high-tier open-source models.
* **👁️ Hybrid Local OCR:** A robust fallback system utilizing `pdfplumber` for digital text and `RapidOCR` (ONNX) with `PyMuPDF` for instantaneous, local extraction of scanned images.
* **🔍 Vector Search Engine:** Built on **FAISS** and **HuggingFace** embeddings for highly accurate, locally processed context retrieval.
* **📊 Smart Quota Engine:** A custom-built global database tracks shared API usage with a slick sidebar progress bar and automatic midnight resets.

---

## 🏗️ The Engineering Blueprint

| Category | Technology |
| --- | --- |
| **Orchestration** | `LangChain`, `Streamlit` |
| **Intelligence** | `Groq API` |
| **Vision & Extraction** | `pdfplumber`, `RapidOCR (ONNX)`, `PyMuPDF` |
| **Search & Retrieval** | `FAISS`, `all-MiniLM-L6-v2` |
| **Security** | `OAuth2`, `JWT`, `CookieController` |

---

## 🎓 Why CleverDocs?

Most RAG apps break the moment they hit a scanned PDF or a cloud rate limit. CleverDocs is built for **resilience**:

1. **Independent Extraction:** Bypasses congested cloud vision servers by running lightweight OCR directly in your environment.
2. **Error-Resilient:** Automatically handles missing cookies and expired OAuth states for a seamless user experience.
3. **Optimized Context:** Uses intelligent text chunking to ensure the AI never misses the "big picture" while staying within processing limits.

---

## 👨‍💻 Developed By

**Abdul Ahad Khan** ([@ahad6510](https://www.google.com/search?q=https://github.com/ahad6510)) with ❤️.

---
