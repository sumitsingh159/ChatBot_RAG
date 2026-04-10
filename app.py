import gradio as gr
import os, io, base64
from PIL import Image
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fitz
from fuzzywuzzy import fuzz

# ------------------ CONFIG ------------------
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_ID = "MBZUAI/LaMini-Flan-T5-248M"

# !! Change this to your name !!
AUTHOR_NAME = "sumitsingh159"

# ------------------ LOAD MODEL ------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = None
current_pdf_path = None

# ------------------ CORE LOGIC ------------------

def upload_file(file):
    global db, current_pdf_path
    if file is None:
        return "❌ No file uploaded."

    file_path = UPLOAD_DIR / "uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(file)

    current_pdf_path = str(file_path)
    loader = PyPDFDirectoryLoader(str(UPLOAD_DIR))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)

    return "✅ Document indexed and ready to query!"


def ask_question(query):
    query_lower = query.lower().strip()
    greetings = ["hi", "hello", "hey", "hola", "who are you", "what can you do"]
    if any(greet in query_lower for greet in greetings):
        if "who are you" in query_lower or "what can you do" in query_lower:
            return "I am your AI PDF Assistant! Upload a document, and I can answer specific questions or summarize parts of it for you.", None
        return "Hello! How can I help you with your document today?", None

    global db
    if db is None:
        return "⚠️ Please upload and index a PDF first.", None

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    context = " ".join([d.page_content.replace('\n', ' ') for d in docs])

    input_text = f" Answer the question \n context: {context} question: {query}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    clean_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if len(clean_answer) < 5:
        clean_answer = "I couldn't find a specific answer. The relevant text found was: " + context[:200] + "..."

    source_img = None
    try:
        if current_pdf_path:
            doc = fitz.open(current_pdf_path)
            best_page, best_score = 0, 0
            for i, page in enumerate(doc):
                score = fuzz.partial_ratio(clean_answer[:100], page.get_text())
                if score > best_score:
                    best_score, best_page = score, i

            pix = doc[best_page].get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            source_img = Image.open(io.BytesIO(pix.tobytes("png")))
            doc.close()
    except:
        pass

    return clean_answer, source_img


# ------------------ CUSTOM CSS ------------------

custom_css = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Design Tokens ── */
:root {
  --bg:        #0b0d11;
  --surface:   #13161d;
  --surface2:  #1a1e28;
  --border:    #252a38;
  --accent:    #5c6eff;
  --accent2:   #ff6b6b;
  --gold:      #f5c842;
  --text:      #e8eaf0;
  --muted:     #6b7280;
  --radius:    14px;
  --glow:      0 0 32px rgba(92,110,255,0.18);
}

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif !important;
  color: var(--text) !important;
  margin: 0 !important;
}

.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 0 24px 60px !important;
}

/* ── Header ── */
#app-header {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 28px 0 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 36px;
}

#app-header::after {
  content: '';
  position: absolute;
  bottom: -1px; left: 0;
  width: 80px; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  border-radius: 2px;
}

.header-left { display: flex; align-items: center; gap: 14px; }

.header-icon {
  width: 46px; height: 46px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  box-shadow: var(--glow);
  flex-shrink: 0;
}

.header-title {
  font-family: 'Syne', sans-serif !important;
  font-size: 26px !important;
  font-weight: 800 !important;
  letter-spacing: -0.5px !important;
  color: var(--text) !important;
  margin: 0 !important;
  line-height: 1 !important;
}

.header-subtitle {
  font-size: 13px !important;
  color: var(--muted) !important;
  margin: 4px 0 0 !important;
  font-weight: 300 !important;
}

.author-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 50px;
  padding: 6px 14px 6px 10px;
  font-size: 13px;
  color: var(--muted);
  font-weight: 400;
  letter-spacing: 0.2px;
  white-space: nowrap;
}

.author-badge .avatar {
  width: 26px; height: 26px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  font-family: 'Syne', sans-serif;
  flex-shrink: 0;
}

.author-badge .by { color: var(--muted); }
.author-badge .name { color: var(--text); font-weight: 500; }

/* ── Section Labels ── */
.section-label {
  font-family: 'Syne', sans-serif !important;
  font-size: 11px !important;
  font-weight: 700 !important;
  letter-spacing: 1.5px !important;
  text-transform: uppercase !important;
  color: var(--accent) !important;
  margin: 0 0 12px !important;
}

/* ── Upload Panel ── */
#upload-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px;
  height: 100%;
  transition: border-color 0.2s;
}
#upload-panel:hover { border-color: rgba(92,110,255,0.35); }

/* ── File Drop Zone ── */
.svelte-1hnfib2, [data-testid="file-upload"], .file-upload-zone {
  background: var(--surface2) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 10px !important;
  transition: all 0.25s !important;
  min-height: 130px !important;
}
.svelte-1hnfib2:hover {
  border-color: var(--accent) !important;
  background: rgba(92,110,255,0.05) !important;
}

/* ── Index Button ── */
#index-btn button, #index-btn {
  background: linear-gradient(135deg, var(--accent), #7b5cf5) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  letter-spacing: 0.3px !important;
  padding: 12px 24px !important;
  cursor: pointer !important;
  width: 100% !important;
  box-shadow: 0 4px 20px rgba(92,110,255,0.3) !important;
  transition: all 0.2s !important;
}
#index-btn button:hover, #index-btn:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 28px rgba(92,110,255,0.45) !important;
}

/* ── Status Box ── */
#status-box textarea, #status-box input {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
}

/* ── Chat Panel ── */
#chat-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px;
  transition: border-color 0.2s;
}
#chat-panel:hover { border-color: rgba(92,110,255,0.35); }

/* ── Query Input ── */
#query-input textarea {
  background: var(--surface2) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 15px !important;
  min-height: 80px !important;
  transition: border-color 0.2s !important;
  resize: vertical !important;
}
#query-input textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(92,110,255,0.12) !important;
  outline: none !important;
}

/* ── Ask Button ── */
#ask-btn button, #ask-btn {
  background: linear-gradient(135deg, #ff6b6b, #ff8e53) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  letter-spacing: 0.3px !important;
  padding: 12px 24px !important;
  cursor: pointer !important;
  width: 100% !important;
  box-shadow: 0 4px 20px rgba(255,107,107,0.28) !important;
  transition: all 0.2s !important;
}
#ask-btn button:hover, #ask-btn:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 28px rgba(255,107,107,0.42) !important;
}

/* ── Answer Box ── */
#answer-box textarea {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 14px !important;
  line-height: 1.7 !important;
  min-height: 120px !important;
}

/* ── Source Image ── */
#source-img {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}

/* ── Tips Panel ── */
#tips-panel {
  background: linear-gradient(135deg, rgba(92,110,255,0.08), rgba(255,107,107,0.05));
  border: 1px solid rgba(92,110,255,0.2);
  border-radius: var(--radius);
  padding: 22px 28px;
  margin-top: 8px;
}

.tip-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 14px;
}

.tip-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
}

.tip-icon { font-size: 20px; margin-bottom: 6px; }
.tip-title {
  font-family: 'Syne', sans-serif;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.5px;
  color: var(--accent);
  margin-bottom: 4px;
}
.tip-desc { font-size: 12px; color: var(--muted); line-height: 1.5; }

/* ── Divider ── */
.divider {
  height: 1px;
  background: var(--border);
  margin: 8px 0 20px;
}

/* ── Labels ── */
label, .label-wrap span {
  color: var(--muted) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 13px !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Footer ── */
#footer {
  text-align: center;
  padding: 28px 0 0;
  color: var(--muted);
  font-size: 12px;
  border-top: 1px solid var(--border);
  margin-top: 40px;
  letter-spacing: 0.2px;
}
#footer span { color: var(--accent2); }

/* ── Animations ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.gradio-container > * {
  animation: fadeInUp 0.5s ease forwards;
}

/* ── Responsive ── */
@media (max-width: 768px) {
  .tip-grid { grid-template-columns: 1fr; }
  .header-title { font-size: 20px !important; }
  .author-badge { display: none; }
}
"""

# ── Author initial (first letter of first word) ──
_initial = AUTHOR_NAME.strip()[0].upper() if AUTHOR_NAME.strip() else "A"

header_html = f"""
<div id="app-header">
  <div class="header-left">
    <div class="header-icon">📄</div>
    <div>
      <div class="header-title">RAG Chatbot</div>
      <div class="header-subtitle">Intelligent PDF Question &amp; Answer</div>
    </div>
  </div>
  <div class="author-badge">
    <div class="avatar">{_initial}</div>
    <span class="by">by</span>
    <span class="name">{AUTHOR_NAME}</span>
  </div>
</div>
"""

tips_html = """
<div id="tips-panel">
  <div class="section-label">💡 Quick Tips</div>
  <div class="tip-grid">
    <div class="tip-card">
      <div class="tip-icon">📂</div>
      <div class="tip-title">Upload</div>
      <div class="tip-desc">Drop any PDF — research papers, manuals, reports, books.</div>
    </div>
    <div class="tip-card">
      <div class="tip-icon">🔍</div>
      <div class="tip-title">Ask</div>
      <div class="tip-desc">Ask specific questions or request a summary of a topic.</div>
    </div>
    <div class="tip-card">
      <div class="tip-icon">🖼️</div>
      <div class="tip-title">Source</div>
      <div class="tip-desc">See the exact PDF page where the answer was found.</div>
    </div>
  </div>
</div>
"""

footer_html = f"""
<div id="footer">
  Built with ❤️ using <span>Gradio</span> · <span>LaMini-Flan-T5</span> · <span>FAISS</span> &nbsp;|&nbsp; Made by <span>{AUTHOR_NAME}</span>
</div>
"""

# ------------------ GRADIO UI ------------------

with gr.Blocks(css=custom_css, title="DocMind AI – PDF Chatbot") as demo:

    # Header
    gr.HTML(header_html)

    # Main layout
    with gr.Row(equal_height=False):

        # ── Left: Upload ──
        with gr.Column(scale=1):
            gr.HTML('<div id="upload-panel">', visible=False)
            gr.HTML('<div class="section-label">📁 Document Upload</div>')
            gr.HTML('<div class="divider"></div>')

            file_input = gr.File(
                label="Drop your PDF here or click to browse",
                type="binary",
                file_types=[".pdf"],
            )

            upload_btn = gr.Button(
                "⚡ Index Document",
                variant="primary",
                elem_id="index-btn",
            )

            status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Waiting for document…",
                elem_id="status-box",
                lines=2,
            )
            gr.HTML('</div>', visible=False)

        # ── Right: Chat ──
        with gr.Column(scale=2):
            gr.HTML('<div id="chat-panel">', visible=False)
            gr.HTML('<div class="section-label">💬 Ask Your Document</div>')
            gr.HTML('<div class="divider"></div>')

            query_input = gr.Textbox(
                label="Your question",
                placeholder="e.g.  What are the key findings?  /  Summarize section 3  /  Who are the authors?",
                lines=3,
                elem_id="query-input",
            )

            ask_btn = gr.Button(
                "🔎 Get Answer",
                variant="primary",
                elem_id="ask-btn",
            )

            with gr.Row():
                with gr.Column(scale=3):
                    output_text = gr.Textbox(
                        label="Answer",
                        lines=6,
                        interactive=False,
                        placeholder="Your answer will appear here…",
                        elem_id="answer-box",
                    )
                with gr.Column(scale=2):
                    output_img = gr.Image(
                        label="Source Page",
                        elem_id="source-img",
                    )

            gr.HTML('</div>', visible=False)

    # Tips row
    gr.HTML(tips_html)

    # Footer
    gr.HTML(footer_html)

    # ── Event Bindings ──
    upload_btn.click(upload_file, inputs=[file_input], outputs=[status])
    ask_btn.click(ask_question, inputs=[query_input], outputs=[output_text, output_img])

demo.launch(theme=gr.themes.Base())