import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import PyPDF2
import io
import requests
import base64
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings

docs=[]
path=""

def update(text,model):
    print(model)
    print(type(model))
    url = "http://127.0.0.1:8003/llm"  # Replace this with the actual FastAPI endpoint
    payload = {"text": text,"model_name" : model}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        image_data = response.json()["image"]
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(image_data)
        # Open image using PIL PIL (Python Imaging Library)
        image = Image.open(io.BytesIO(image_bytes))
        # You can display the image or do further processing here
        return response.json()["Output"], image
    else:
        return "Error: Failed to process the request."




i_Number=0
def upload_file(file):
    url = "http://127.0.0.1:8003/upload"  # Replace this with your API endpoint
    global i_Number
    filename_pdf = f"document{i_Number}.pdf"
    i_Number+=1
    files = {'file': (filename_pdf, file, 'application/pdf')} #This line prepares the file to be uploaded. It creates a dictionary called files where the key is 'file' (which corresponds to the field name expected by the API endpoint), and the value is a tuple containing the filename, the file object itself, and the MIME type of the file, which in this case is 'application/pdf'.

    headers = {'accept': 'application/json'} #This line creates a dictionary called headers containing the HTTP headers to be included in the request. Here, it specifies that the client expects a JSON response.
    response = requests.post(url, files=files, headers=headers)
    return response.text


io2 =  gr.Interface(
        fn=update,
        inputs=[gr.Textbox(label="Ask Question",placeholder="What is your Question?"),
                    gr.Dropdown(label="Select LLM Model", choices=["google/gemma-2b-it", "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3-8B"])],
        # outputs=gr.Textbox(),
        outputs=[gr.Textbox(label="Answer"), gr.Image(label="Image Reference of the answer")],
        # title="Name Interface",
        # theme="compact",
        allow_flagging = 'never'
    )
io1 = gr.Interface(
    fn=upload_file,
    inputs=gr.File(label="Upload PDF", type="binary"),
    outputs="text",
    title="PDF Uploader",
    description="Upload a PDF document to a FastAPI server.",
    allow_flagging='never'
)


def show_row(value):
    if value=="Upload Document":
        return gr.update(visible=True), gr.update(visible=False)
    if value=="LLMs":
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False)


with gr.Blocks() as demo:
    d = gr.Dropdown(label="Select Interface", choices=["Upload Document", "LLMs"])
    with gr.Column(visible=False) as r1:
        io1.render()
    with gr.Row(visible=False) as r2:
        io2.render()
        # io3.render()
    d.change(show_row, d, [r1, r2])

demo.launch()
