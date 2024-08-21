# llm-pdf-chatbot

## 실행방법
```bashscript
$ python3 -m pip install -r requirements.txt // 의존성 설치
$ streamlit run pdfgpt.py // 실행
```

## 실행화면설명
<img width="1722" alt="image" src="https://github.com/user-attachments/assets/ad351c69-7baf-4f4e-84d9-46d17f0ba413">

1. LLM 선택
    - gpt 4 버전의 LLM model 선택 가능
2. OpenAI API Key
    - openai api key를 넣어주세요(안 넣으면 실행되지 않음)
3. pdf 파일 업로드
    - pdf 형식의 파일만 업로드 가능
4. Process 버튼
    - Process를 누르면 embeding 시작 오른쪽 상단의 Running이 끝날때까지 대기
5. 대화초기화
    - 대화를 초기화 합니다

<img width="1696" alt="image" src="https://github.com/user-attachments/assets/0d191e58-7c3d-4948-a073-787babd4832c">

질문을 하게 되면 문서에 있는 내용으로 대답을 해줍니다

## 개선해야할 사항
1. 멀티턴을 구현하려고 히스토리를 구현했지만, 정확한 이전 데이터를 인식하지 못하여서 멀티턴 구현 고도화 필요
2. pdf 업로드 시 시간이 오래걸리고, 동기 방식이라 사용자 UX가 좋지 않아서 고도화 필요
3. Vectorstore를 pinecone을 써보고 싶었으나 개발 시간 한계로 인해 추후 고도화 필요
4. model도 openai말고 여러개 사용을 하고 싶었으나 개발 시간 한계로 인해 추후 고도화 필요

