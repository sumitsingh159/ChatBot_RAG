
import base64
import shutil

from fastapi import FastAPI, File, UploadFile
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
import torch
from pdf2image import convert_from_path
from fastapi import FastAPI, File, UploadFile, HTTPException
from pdf2image import convert_from_path
from PIL import Image
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import fitz
from fuzzywuzzy import fuzz

app = FastAPI()
db=None
docs=None
strings = []
# Create directory if it doesn't exist
UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file):

    global strings
    strings.append("./"+str(UPLOAD_DIR / uploaded_file.filename))

    with open(UPLOAD_DIR / uploaded_file.filename, "wb") as buffer: #This line opens a file for writing in binary mode ("wb") using a context manager (with statement). The file name is constructed using the UPLOAD_DIR path (which presumably is a directory path) and the filename attribute of the uploaded_file object. This line prepares a file in the specified directory to save the uploaded content.

        shutil.copyfileobj(uploaded_file.file, buffer) #This line copies the content of the uploaded file (uploaded_file.file) to the opened file (buffer). It utilizes the shutil.copyfileobj() function from the shutil module. This effectively writes the content of the uploaded file to the file opened for writing.

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global db
    global docs
    save_uploaded_file(file)

    loader = PyPDFDirectoryLoader("./uploaded_files", extract_images=True)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    chunked_docs = splitter.split_documents(docs)


    db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"))

    return {"filename": file.filename, "Successfully": "Uploaded"}

def get_pdf_page_ss(page_number_pdf,pdf_path):
    pdf_file=pdf_path
    images = convert_from_path(pdf_file, first_page=page_number_pdf, last_page=page_number_pdf)
    if images:
            # Convert the PIL image to bytes
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
            # Convert bytes to base64 string
        image_base64 = base64.b64encode(img_byte_arr).decode('utf-8') #This line encodes the image data (stored in img_byte_arr) into a base64-encoded string using the base64.b64encode() function. The resulting byte data is then decoded into a UTF-8 string using .decode('utf-8'). The variable image_base64 now contains the base64-encoded string representing the image.
        return image_base64

def Image_from_pdf(search_text):

    global strings
    for i in range(len(strings)):


        with fitz.open(strings[i]) as pdf_doc:
            # Iterate over each page in the document
            max_similarity = 0
            most_similar_page = None
            pdf_path_for_answer=None

            # Iterate over each page in the document
            for page_number in range(len(pdf_doc)):
                # Get the text of the current page
                page = pdf_doc.load_page(page_number)
                text = page.get_text()

                # Calculate similarity between page text and search text
                similarity = fuzz.partial_ratio(search_text, text)

                # Update max similarity and most similar page if needed
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_page = page_number + 1  # Add 1 to convert to 1-based indexing
                    pdf_path_for_answer = strings[i]

    print(most_similar_page, max_similarity,pdf_path_for_answer)
    return get_pdf_page_ss(most_similar_page,pdf_path_for_answer)

class TextRequest(BaseModel):
    text: str
    model_name: str

@app.post("/llm")
async def llm(request: TextRequest):
    try:
        text1=request.text
        model_name1 = request.model_name

        global db
        print(db)
        global docs

        model_name=model_name1
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True, )

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1024,
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        prompt_template = """
        <|system|>
        Answer the question based on your knowledge. Use the following context to help:

        {context}

        </s>
        <|user|>
        {question}
        </s>
        <|assistant|>

         """
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        llm_chain = prompt | llm | StrOutputParser()  # The | operator likely represents composition or chaining, where the output of one component is passed as input to the next. So, the prompt is first generated using the template, then passed through the language model, and finally parsed to extract the desired output.


        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
        output = rag_chain.invoke(text1)

        split_result = output.split("<|assistant|>", 1)
        resultss=split_result[1].strip()
        search_text = resultss
        global strings

        image=Image_from_pdf(search_text)

        return {"Output": resultss,"image" : image}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


