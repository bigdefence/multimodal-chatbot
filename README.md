# 🤖 멀티모달 한국어 챗봇 프로젝트

## 📝 프로젝트 소개

이 프로젝트는 음성 인식, 텍스트 대화, 이미지 생성 기능을 통합한 멀티모달 한국어 챗봇 시스템입니다. 사용자는 음성이나 텍스트로 질문할 수 있으며, 챗봇은 텍스트로 응답하고 필요시 이미지를 생성할 수 있습니다.

## ✨ 주요 기능

- 🎤 음성 인식: 사용자의 음성 입력을 텍스트로 변환
- 💬 텍스트 대화: LLaMA 3 기반의 한국어 대화 모델
- 🖼️ 이미지 생성: Stable Diffusion 3를 이용한 이미지 생성
- 🔊 음성 합성: 챗봇 응답을 음성으로 재생

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **AI 모델**:
  - 대화: MLP-KTLim/llama-3-Korean-Bllossom-8B
  - 음성 인식: Whisper (large-v3)
  - 이미지 생성: Stable Diffusion 3

## 🚀 설치 및 실행 방법

1. 저장소 클론:
   ```
   git clone [저장소 URL]
   cd [프로젝트 디렉토리]
   ```

2. 필요한 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

3. 각 서비스 실행:
   - 챗봇 서버:
     ```
     python fastapi_chatbot.py
     ```
   - 이미지 생성 서버:
     ```
     python stablediffusion.py
     ```
   - 프론트엔드:
     ```
     streamlit run app1.py
     ```

## 📖 사용 방법

1. Streamlit 앱 접속
2. '음성 녹음' 버튼을 클릭하여 질문을 음성으로 녹음 (5초)
3. 또는 채팅 입력창에 텍스트로 질문 입력
4. '그림그려줘'와 같은 문구 포함 시 이미지 생성
5. AI의 응답을 텍스트로 확인하고 음성으로 듣기

## 👨‍💻 개발자 정보

- **개발자**: 정강빈
- **버전**: 2.0.0

## 🔗 API 엔드포인트

- 챗봇 API: `챗봇 토큰`
- 이미지 생성 API: `이미지 토큰/generate_image`

## 📜 라이선스

이 프로젝트는 [라이선스 이름] 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

---

❗ **참고**: API 엔드포인트는 개발 환경에 따라 변경될 수 있습니다. 실제 배포 시 안정적인 URL로 업데이트해주세요.
