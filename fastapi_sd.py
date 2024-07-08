
import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from io import BytesIO
import base64
from pyngrok import ngrok
import uvicorn
import nest_asyncio

login('허깅페이스 토큰')

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

app = FastAPI()

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    num_inference_steps: int = 28
    guidance_scale: float = 7.0

@app.post("/generate_image")
async def generate_image(request: ImageRequest):
    try:
        image = pipe(
            request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
        ).images[0]

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    nest_asyncio.apply()

    ngrok.set_auth_token("ngrok 토큰")
    public_url = ngrok.connect(8000)
    print(f"Public URL: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)

