# 📄 AI PDF Chatbot (RAG Assistant)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/sumitsingh159/RAG_CHATBOT)

**Author:** [Sumit Singh](https://huggingface.co/sumitsingh159)

An intelligent, world-class Document AI assistant that allows you to chat with your PDF files. This application uses **Retrieval-Augmented Generation (RAG)** to provide precise answers based only on the content of your uploaded documents.

---

## 🚀 Live Demo
Access the live application here:  
👉 **[https://huggingface.co/spaces/sumitsingh159/RAG_CHATBOT](https://huggingface.co/spaces/sumitsingh159/RAG_CHATBOT)**

---

## ✨ Features
* **Intelligent QA:** Ask complex questions about your PDF and get concise, accurate answers.
* **Visual Citations:** The AI doesn't just tell you the answer; it shows you the exact page from the PDF where it found the information.
* **Conversational Router:** Handles greetings (Hi, Hello) naturally before switching to document analysis.
* **Privacy-Focused:** Built with open-source models (**LaMini-T5** and **BGE Embeddings**) that can run entirely offline.
* **Glassmorphism UI:** A premium, modern interface built with Gradio.

---

## 🛠️ Technology Stack
This project leverages the latest in AI and Vector search:

* **LLM:** `LaMini-Flan-T5-248M` (Optimized for instruction following).
* **Vector Database:** `FAISS` (Facebook AI Similarity Search).
* **Embeddings:** `BGE-base-en-v1.5` (State-of-the-art text embeddings).
* **Orchestration:** `LangChain` & `Hugging Face Pipelines`.
* **Frontend:** `Gradio` (Modern Web Interface).
* **PDF Engine:** `PyMuPDF` (fitz) & `PyPDF`.

---

## 📖 How It Works


1.  **Upload:** You upload a PDF document.
2.  **Indexing:** The app splits the PDF into small chunks and converts them into mathematical vectors (embeddings).
3.  **Search:** When you ask a question, the app finds the top 5 most relevant chunks using FAISS.
4.  **Generation:** The AI model reads those chunks and your question to generate a human-like response.
5.  **Source Mapping:** The app uses fuzzy matching to find the specific page in your PDF and renders it for your reference.

---
