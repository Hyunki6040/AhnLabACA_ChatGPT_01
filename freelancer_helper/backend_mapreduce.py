


"""
파이썬으로 백엔드 서버 프로그램을 만드는 중.

각각 uri 별로 request 값과 result 값이 아래와 같은 서버 프로그램 코드를 작성하고  스웨거를 적용시켜줘.
flask-restx 를 사용 할 것

/new_token

request : {
  db : integer
}
result : {
  token: string
}


/prompt

request : {
  token: string
  prompt: string
}

result : {
  result: string
}

"""
import threading
import uuid
import asyncio
import os
import time
import json
import sys
import langchain
import openai

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Iterable, List
from langchain.docstore.document import Document

from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.schema.vectorstore import VectorStore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

sys.path.append(os.getenv("PYTHONPATH"))

from utils import (
  BusyIndicator,
  ConsoleInput,
  get_filename_without_extension,
  load_pdf_vectordb,
  load_vectordb_from_file,
  get_vectordb_path_by_file_path
  )
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")

llm_model = "gpt-3.5-turbo"
PDF_FILE = "./data/프리랜서 가이드라인 (출판본).pdf"
llm = ChatOpenAI(model_name=llm_model, temperature=0)

is_debug = True
app = FastAPI(debug=is_debug, docs_url="/api-docs")


class TokenOutput(BaseModel):
  token: str


class PromptRequest(BaseModel):
  token: str
  prompt: str


class PromptResult(BaseModel):
  result: str

def check_and_create_vectorDB():
  file = PDF_FILE
  busy_indicator = BusyIndicator.busy(True, f"{get_filename_without_extension(file)} db를 로딩 중입니다 ")
  vectordb : FAISS = load_vectordb_from_file(file)
  busy_indicator.stop()
  return vectordb
  

# @app.get("/")
# async def serve_html():
#   return FileResponse('./html-docs/index.html')

tokens = {

}
@app.get("/api/new_token")
async def new_token(db: int):
  # 원하는 db 처리 로직을 여기에 추가하실 수 있습니다.
  token=str(uuid.uuid4())
  vectordb = check_and_create_vectorDB()
  _template = """Given the following conversation and a follow up \
  question, rephrase the follow up question to be a standalone \
  question, in its original language. 
  
  Chat History:
  {chat_history}
  Follow Up Input: {question}
  Standalone question:"""
  CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
  memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key='answer',
    return_messages=True
  )
  retriever=vectordb.as_retriever()
  executor = ConversationalRetrievalChain.from_llm(
      llm,
      retriever=retriever,
      return_source_documents=True,
      return_generated_question=True,
      max_tokens_limit=4097,
      memory=memory,
      # 추가된 영역
      condense_question_prompt =  CONDENSE_QUESTION_PROMPT
  )
  tokens[token] = executor
  return jsonable_encoder(TokenOutput(token=token))

request_idx = 0

@app.post("/api/prompt")
async def process_prompt(request: PromptRequest):
  # 비동기적으로 처리할 내용을 여기에 구현합니다.
  # 예를 들어, 외부 API 호출이나 무거운 계산 작업 등을 비동기로 수행할 수 있습니다.
  
  busy_indicator = BusyIndicator().busy(True)
  executor = tokens[request.token]
  if not executor:
    raise ValueError("token이 없습니다.")
  result = executor({"question": request.prompt})
  busy_indicator.stop()
  page_nums = []
  
  pages = ""
  docs = result['source_documents']
  if docs != None:
    # p = lambda meta, key: print(f"{key}: {meta[key]}") if key in meta else None
    p = lambda meta, key: (f"{meta[key]}{key}") if key in meta else None
    for doc in docs:
      page_nums.append(p(doc.metadata, 'page'))
    pages = str("\n\n[%s]" % ", ".join(page_nums))
  
  return jsonable_encoder(PromptResult(result=result['answer']+pages))


# app.mount("/", StaticFiles(directory="AhnLabACA_ChatGPT_01/html-docs", html=True), name="static")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=5000)
