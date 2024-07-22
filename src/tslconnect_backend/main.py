import os

import aiofiles
import torch
from fastapi import FastAPI, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import PreTrainedTokenizerFast

from .model import Sign2ThaiT5, Thai2SignT5
from .utils import (
    extract_keypoints,
    forward_fill_landmarks,
    normalize_landmarks_scale,
    pad_landmarks,
)
from .video_utils import make_video_from_words

MAX_FRAMES = 512
MAX_INPUT = 512

RESUT_LINK = "https://api.mystrokeapi.uk/video/"

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokenizer
thai_tokenizer = PreTrainedTokenizerFast(tokenizer_file="thai_tokenizer.json")
thai_vocab_size = thai_tokenizer.vocab_size

sign_tokenizer = PreTrainedTokenizerFast(tokenizer_file="sign_tokenizer.json")
sign_vocab_size = sign_tokenizer.vocab_size

# Load model
sign2thai_model = Sign2ThaiT5(vocab_size=thai_vocab_size)
sign2thai_model.load_state_dict(torch.load("sign2thai_model.bin"))
sign2thai_model.to(device)

thai2sign_model = Thai2SignT5(
    vocab_size=thai_vocab_size, sign_vocab_size=sign_vocab_size
)
thai2sign_model.load_state_dict(torch.load("thai2sign_model.bin"))
thai2sign_model.to(device)


# Inference
def sign2thai_inference(video_path):
    landmarks = extract_keypoints(video_path)
    landmarks = forward_fill_landmarks(landmarks)
    landmarks = normalize_landmarks_scale(landmarks)
    landmarks_pad = pad_landmarks(
        torch.tensor(landmarks, dtype=torch.float32), max_frames=MAX_FRAMES
    )

    with torch.no_grad():
        outputs = sign2thai_model.generate(
            landmarks_pad[0].unsqueeze(0).to(device),
            landmarks_pad[1].unsqueeze(0).to(device),
            max_length=128,
        )

    thai_text = thai_tokenizer.decode(outputs[0], skip_special_tokens=True)
    thai_text = thai_text.replace(" ", "")

    return thai_text


def thai2sign_inference(thai_text):
    with torch.no_grad():
        inputs = thai_tokenizer(
            thai_text, return_tensors="pt", padding="max_length", max_length=MAX_INPUT
        )
        outputs = thai2sign_model.generate(
            inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=128,
        )

    sign_text = sign_tokenizer.decode(outputs[0], skip_special_tokens=True)
    words = sign_text.split()

    video_name = make_video_from_words(words)

    return video_name


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/sign2thai")
async def sign2thai(file: UploadFile):
    try:
        async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as temp:
            try:
                content = await file.read()
                await temp.write(content)
            except Exception as e:
                return {"message": str(e)}
            finally:
                await file.close()

        output = await run_in_threadpool(sign2thai_inference, temp.name)
        response = {"translation": output}
    except Exception as e:
        return {"message": str(e)}
    finally:
        os.remove(temp.name)

    return response

@app.post("/thai2sign")
async def thai2sign(text: str):
    output = await run_in_threadpool(thai2sign_inference, text)
    response = {"link": f"{RESUT_LINK}{output}"}
    return response

@app.get("/video/{video_name}")
async def get_video(video_name: str):
    return FileResponse(f"results/{video_name}", media_type='video/mp4')
