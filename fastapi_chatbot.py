
from transformers import AutoTokenizer, AutoModelForCausalLM
from faster_whisper import WhisperModel
import torch
chat_model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_id)
chat_model = AutoModelForCausalLM.from_pretrained(
    chat_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
chat_model.eval()

whisper_model_size = "large-v3"
whisper_model = WhisperModel(whisper_model_size, device="cuda", compute_type="float16")

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from pyngrok import ngrok
import nest_asyncio
import uvicorn
import io

app = FastAPI()


class ChatInput(BaseModel):
    message: str

@app.post("/chat")
async def chat(chat_input: ChatInput):
    PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": chat_input.message}
    ]

    input_ids = chat_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(chat_model.device)

    terminators = [
        chat_tokenizer.eos_token_id,
        chat_tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = chat_model.generate(
            input_ids,
            max_new_tokens=400,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

    response = chat_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return {"response": response}

@app.post("/speech_to_chat")
async def speech_to_chat(file: UploadFile = File(...)):
    contents = await file.read()
    audio = io.BytesIO(contents)

    segments, _ = whisper_model.transcribe(audio, language="ko")
    transcribed_text = " ".join([segment.text for segment in segments])
del
    chat_input = ChatInput(message=transcribed_text)
    chat_response = await chat(chat_input)

    return {"transcription": transcribed_text, "response": chat_response["response"]}

if __name__ == "__main__":
    nest_asyncio.apply()

    ngrok.set_auth_token("ngrok 토큰")
    public_url = ngrok.connect(8001)
    print(f"Public URL: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=8001)

