import streamlit as st
import requests
import io
import sounddevice as sd
import scipy.io.wavfile as wav
import threading
import time
from gtts import gTTS
import os
import pygame
import base64
from PIL import Image

st.set_page_config(page_title="Multi-Modal Korean ChatBot", layout="wide")

# API 엔드포인트 설정
CHAT_API_ENDPOINT = "chat_api"
IMAGE_API_ENDPOINT = "stable_diffusion_api/generate_image"

st.sidebar.title("챗봇 정보")
st.sidebar.info(
    "이 챗봇은 음성 인식, Llama3 모델, 그리고 이미지 생성 기능을 통합한 시스템입니다. "
    "음성으로 질문하거나 텍스트로 입력할 수 있으며, '그림그려줘'라고 요청하면 이미지를 생성할 수 있습니다."
)

st.sidebar.title("사용 방법")
st.sidebar.markdown(
    """
    1. '음성 녹음' 버튼을 클릭하여 5초 동안 음성을 녹음하세요.
    2. 녹음이 완료되면 자동으로 서버로 전송됩니다.
    3. 또는 채팅 입력창에 텍스트로 질문을 입력하세요.
    4. '그림그려줘'와 같은 문구를 포함하면 이미지가 생성됩니다.
    5. AI가 응답을 생성하고 음성으로 읽어줍니다.
    6. 대화 기록은 자동으로 저장됩니다.
    """
)

def record_audio(duration=5, sample_rate=16000):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio_data, sample_rate

def send_audio_get_response(audio_data, sample_rate):
    audio_buffer = io.BytesIO()
    wav.write(audio_buffer, sample_rate, audio_data)
    audio_buffer.seek(0)
    
    files = {'file': ('audio.wav', audio_buffer, 'audio/wav')}
    response = requests.post(f"{CHAT_API_ENDPOINT}/speech_to_chat", files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"transcription": "Error: Unable to process audio.", "response": "Error: Unable to get response from the server."}

def send_text_get_response(text):
    response = requests.post(f"{CHAT_API_ENDPOINT}/chat", json={"message": text})
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Error: Unable to get response from the server."

def text_to_speech(text):
    tts = gTTS(text=text, lang='ko')
    tts.save("response.mp3")
    
    pygame.mixer.init()
    pygame.mixer.music.load("response.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    
    os.remove("response.mp3")

def generate_image(prompt):
    response = requests.post(IMAGE_API_ENDPOINT, json={
        "prompt": prompt,
        "negative_prompt": "",
        "num_inference_steps": 28,
        "guidance_scale": 7.0
    })
    if response.status_code == 200:
        image_data = base64.b64decode(response.json()["image"])
        return Image.open(io.BytesIO(image_data))
    else:
        st.error(f"Error: {response.status_code}")
        return None

def record_and_send():
    with st.spinner("음성을 녹음 중..."):
        audio_data, sample_rate = record_audio()
    
    with st.spinner("음성을 처리 중..."):
        result = send_audio_get_response(audio_data, sample_rate)
    
    transcription = result['transcription']
    if "그림" in transcription:
        with st.spinner("이미지 생성 중..."):
            image = generate_image(transcription)
            if image:
                st.session_state.messages.append({"role": "assistant", "image": image})
                st.image(image)
                response = "이미지가 생성되었습니다."
            else:
                response = "이미지 생성에 실패했습니다."
    else:
        response = result['response']
    
    st.session_state.messages.append({'role': 'user', 'content': f"{transcription}"})
    st.session_state.messages.append({'role': 'assistant', 'content': response})
    
    text_to_speech(response)

def main():
    st.title('🤖 Multimodal ChatBot - 음성, 텍스트, 이미지 통합 AI')
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            if "content" in message:
                st.markdown(message['content'])
            if "image" in message:
                st.image(message['image'])

    if st.sidebar.button("음성 녹음"):
        record_and_send()
        st.experimental_rerun()

    st.sidebar.title("개발자 정보")
    st.sidebar.markdown(
        """
        - **개발자**: 정강빈
        - **버전**: 2.0.0
        """
    )

    if prompt := st.chat_input('질문을 입력하세요'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            if "그림그려줘" in prompt or "이미지 생성" in prompt:
                with st.spinner("이미지 생성 중..."):
                    image = generate_image(prompt)
                    if image:
                        st.session_state.messages.append({"role": "assistant", "image": image})
                        st.image(image)
                        response = "이미지가 생성되었습니다."
                    else:
                        response = "이미지 생성에 실패했습니다."
            else:
                with st.spinner('답변 생성 중...'):
                    response = send_text_get_response(prompt)
            
            st.markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
            text_to_speech(response)

if __name__ == "__main__":
    main()
