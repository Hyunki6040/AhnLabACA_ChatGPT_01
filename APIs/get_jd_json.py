import json
import sys
import os
import threading

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import asyncio

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import openai
from dotenv import load_dotenv

load_dotenv()

is_debug = True
app = FastAPI(debug=is_debug, docs_url="/api-docs")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION")


class jdRequest(BaseModel):
  url: str



@app.post("/get_jd_json")
async def get_jd_json(request: jdRequest):
    # 본문을 Chunk 단위로 쪼갬
    text_splitter = CharacterTextSplitter(        
        separator="\n\n",
        chunk_size=3000,     # 쪼개는 글자수
        chunk_overlap=300,   # 오버랩 글자수
        length_function=len,
        is_separator_regex=False,
    )

    # 웹사이트 내용 크롤링 후 Chunk 단위로 분할
    docs = WebBaseLoader(request.url).load_and_spl0it(text_splitter)

    # 각 Chunk 단위의 템플릿
    template = '''채용공고: {text}

    채용공고에서 다음과 같은 정보를 추출해. 정보가 없다면 "NULL"으로 표시해:
    {format_instructions}
    '''

    # 전체 문서(혹은 전체 Chunk)에 대한 지시(instruct) 정의
    combine_template = '''채용공고: {text}

    채용공고에서 NULL은 비어있는 정보야. 채용공고에서 다음과 같은 정보를 추출해. 정보가 없다면 "수집중.."으로 표시해:
    {format_instructions}
    '''

    # LLM 객체 생성
    llm = ChatOpenAI(temperature=0,
                    model_name='gpt-3.5-turbo-16k')

    # 추출할 정보의 정의
    response_schemas = [
        ResponseSchema(name="job_title", description="직무명을 반드시 포함한 채용공고의 제목. 15자이내로 작성하고 '(D-)'는 포함하지마."),
        ResponseSchema(name="company_name", description="채용공고에 작성된 회사의 이름"),
        ResponseSchema(name="team_name", description="채용공고에 작성된 팀의 이름"),
        ResponseSchema(name="keyword", description="채용공고에 작성된 업계와 직무에 대해 유사하고 관련한 키워드들을 list 형태로 추출. 키워드의 예시로는 데이팅앱, 매칭, IT, 소셜 매칭, 플랫폼, 엔터, 크리에이터, 팬덤 플랫폼, 콘텐츠, 상품, 온오프라인 통합, 커뮤니티, 콘텐츠 기획제작, 크리에이터 B2B, 병의원, 치의과, 기공소,  B2B, 글로벌, B2B, 초기전략, 웹툰, 콘텐츠, 글로벌, 이커머스, 서비스 기획, Web, App, Admin, 동남아시아, 중동아시아, 여성유저, 심리상담, 콘텐츠, 병의원, 멘탈헬스케어, 디지털마케팅, google, meta, 심리상담, 콘텐츠 등과 같이 직무 및 업계에 관한 단어들"),
        ResponseSchema(name="process", description="채용공고에 작성된 지원 절차를 불렛포인트 형식으로 작성"),
        ResponseSchema(name="documents", description="채용공고에 작성된 제출 필요 지원 서류들을 list 형태로 추출. 지원 서류의 예시로는 경력기술서, 이력서, 포트폴리오, 자기소개서, 직무 과제"),
        ResponseSchema(name="main_duty", description="채용공고에 작성된 해당 직무의 주요업무를 불렛포인트 형식으로 작성"),
        ResponseSchema(name="qualifications", description="채용공고에 작성된 자격요건을 불렛포인트 형식으로 작성"),
        ResponseSchema(name="preferred", description="채용공고에 작성된 우대사항을 불렛포인트 형식으로 작성"),
        ResponseSchema(name="company_profile", description="채용공고에 작성된 회사와 팀에 대한 소개의 주요내용을 불렛포인트 형식으로 작성"),
        ResponseSchema(name="welfare", description="채용공고에 작성된 사내 혜택 및 복지 정보을 불렛포인트 형식으로 작성"),
        ResponseSchema(name="work_location", description="채용공고에 작성된 근무 위치 작성"),
        ResponseSchema(name="comment", description="채용공고의 지원자격과 주요업무를 분석해 어떤 이직자에게 추천하는 지 소개하는 1줄짜리 코멘트를 창의적으로 작성"),
    ]

    # 출력 파서 생성
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    # 템플릿 생성
    prompt = PromptTemplate(template=template, input_variables=['text'], partial_variables={"format_instructions": format_instructions})
    combine_prompt = PromptTemplate(template=combine_template, input_variables=['text'], partial_variables={"format_instructions": format_instructions})

    # 요약을 도와주는 load_summarize_chain
    summarize_chain = load_summarize_chain(llm, 
                                map_prompt=prompt, 
                                combine_prompt=combine_prompt,
                                chain_type="map_reduce", 
                                verbose=False)

    # 요약 결과 추출
    # response = summarize_chain.run(docs, format_instructions=format_instructions)
    response = summarize_chain.run(docs)

    # JSON으로 출력
    return JSONResponse(content=parser.parse(response))


if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=5000)