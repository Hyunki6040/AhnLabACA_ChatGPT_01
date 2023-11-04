# AhnLabACA_ChatGPT_01
이 소스코드에 대한 README.md 파일은 다음과 같이 작성할 수 있습니다.

# Freelancer Helper

Freelancer Helper는 python에서 동작하는 프리랜서 가이드라인 (출판본).pdf 파일을 대화형으로 검색하고 답변하는 소스코드입니다. "freelancer_helper" 폴더의 backend.py와 frontend.py 파일이 핵심입니다.

## backend.py

backend.py는 FAST API 기반으로 작성되었습니다. backend.py에서 다음과 같은 API를 제공합니다.

- `/api/new_token`: GET 방식으로 호출하면, "프리랜서 가이드라인 (출판본).pdf" 파일을 "FAISS.from_documents" 함수를 이용해서 vectorDB로 변환합니다. 그리고 변환된 vectorDB를 vectordb 폴더에 저장합니다. 저장된 vectorDB로 "ConversationalRetrievalChain.from_llm" 함수를 이용해 condense_question_prompt를 포함한 대화형 chain을 생성합니다. 또한 token을 생성하고 token을 기반으로 동일한 token으로 접근하면 동일한 chain으로 대화를 이어서 할 수 있도록 token을 저장합니다. 이 API는 생성된 token을 JSON 형식으로 응답합니다.
- `/api/prompt`: POST 방식으로 PromptRequest라는 class를 request로 보내면, 대화형 chain을 통해 PDF 파일에 내용과 참조 페이지를 답변으로 응답합니다. PromptRequest라는 class는 token(str)과 prompt(str)로 구성되어 있습니다. 이 API는 답변과 페이지를 JSON 형식으로 응답합니다.

## frontend.py

frontend.py는 CLI 환경에서 실행 됩니다. frontend.py를 실행하면, backend.py의 `/api/new_token` API를 호출하여 token을 받아옵니다. 그리고 사용자에게 prompt를 입력받아, backend.py의 `/api/prompt` API를 호출하여 답변과 페이지를 받아옵니다. 이 과정을 반복하여 사용자와 대화를 이어갑니다. frontend.py는 PDF 파일에 내용과 gpt-3.5-turbo를 기반으로 대화할 수 있습니다.

## 실행 방법

이 소스코드를 실행하는 방법은 다음과 같습니다.

1. python 3.11.6 버전에서 .venv를 설정하고 접속합니다.
2. mac 기준, requirements_linux.txt 파일안에 명시 된 라이브러리를 설치합니다. .venv안에서 다음의 명령어로 설치할 수 있습니다.

```bash
pip install -r requirements_linux.txt
```

3. .venv안에서 다음의 명령어로 backend.py를 실행하면 backend가 실행됩니다.

```bash
python freelancer_helper/backend.py
```

4. .venv안에서 다음의 명령어로 frontend.py를 실행하면 frontend가 실행됩니다.

```bash
python freelancer_helper/frontend.py
```

5. frontend에서 prompt를 입력하고 엔터를 누르면, 답변과 페이지를 받아볼 수 있습니다. 대화를 종료하려면, "q" 또는 "Q" 또는 "ㅂ"를 입력하고 엔터를 누르면 됩니다.