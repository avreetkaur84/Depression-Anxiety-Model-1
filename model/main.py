from fastapi import FastAPI
from pydantic import BaseModel
from model.predictor import predict_transcript
from fastapi.middleware.cors import CORSMiddleware

import nltk
nltk.download('punkt')

app = FastAPI()

# ✅ Allow all origins (for React/Next frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request schema
class TranscriptRequest(BaseModel):
    transcript: str

# ✅ Route
@app.post("/predict")
async def predict(data: TranscriptRequest):
    prediction = predict_transcript(data.transcript)
    return prediction