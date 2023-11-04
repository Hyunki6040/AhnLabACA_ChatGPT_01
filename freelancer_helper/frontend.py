import requests

"""
ChatGPT 프롬프트
아래의 back-end 서버의 API에 접속하는 파이썬용 함수 만들어줘.
함수를 클래스용으로 바꿔줘.
"""


class APIClient:
  def __init__(self, base_url):
    """
    서버 클라이언트 클래스 생성자

    :param base_url: 서버의 기본 URL (예: 'http://localhost:5000')
    """
    self.base_url = base_url

  def get_new_token(self, db_value):
    """
    서버의 /new_token 엔드포인트에 GET 요청을 보내서 새로운 토큰을 가져옵니다.

    :param db_value: 데이터베이스 식별자
    :return: 받아온 토큰 문자열
    """
    response = requests.get(f"{self.base_url}/api/new_token", params={'db': db_value})
    data = response.json()
    return data['token']

  def send_prompt(self, token, prompt_text):
    """
    서버의 /prompt 엔드포인트에 POST 요청을 보내서 결과를 가져옵니다.

    :param token: 사용할 토큰
    :param prompt_text: 사용자의 입력 텍스트
    :return: 서버에서 처리된 결과 문자열
    """
    payload = {
      'token': token,
      'prompt': prompt_text
    }
    response = requests.post(f"{self.base_url}/api/prompt", json=payload)
    data = response.json()
    return data['result']


class CmdInterface:

  def connect(self, base_url: str, db: int) -> str:
    self.connector = APIClient(base_url)
    self.token = self.connector.get_new_token(db)
    return self.token

  def prompt(self, prompt_text: str) -> str:
    if not self.connector:
      raise ValueError("connector가 설정되어 있지 않습니다.")
    if not self.token :
      raise ValueError("token 값이 없습니다.")
    return self.connector.send_prompt(self.token, prompt_text)


def main():
  base_url = 'http://localhost:5000'  # 서버의 주소
  db_value = 1  # 예시 데이터베이스 식별자
  cli = CmdInterface()
  cli.connect(base_url, db_value)

  while True:
    prompt = input("prompt >> ")

    if prompt == '':  # 빈 라인인 경우.
      continue

    if prompt == 'q' or prompt == 'Q' or prompt == 'ㅂ':
      break
    
    answer = cli.prompt(prompt)
    print(f"answer : {answer}")



if __name__ == '__main__':
  main()
