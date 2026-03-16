import json
import os
import tempfile
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
client = OpenAI()


class AnalyzeRequest(BaseModel):
    transcription: str


@app.get("/")
def read_root() -> Dict[str, str]:
    return {"status": "InsurCall Sentinel listening"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file uploaded.",
        )

    _, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".wav"

    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_path = temp_file.name

        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        with open(temp_path, "rb") as audio_file:
            try:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )
            except Exception as exc:  # pragma: no cover - external dependency
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Transcription service error: {exc}",
                ) from exc

        text = getattr(transcription, "text", None) or ""
        if not text:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Empty transcription received from service.",
            )

        return {"transcription": text}
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error while processing audio: {exc}",
        ) from exc
    finally:
        try:
            if "temp_path" in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass


@app.post("/analyze")
async def analyze_transcription(payload: AnalyzeRequest) -> Dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert insurance fraud detection analyst. "
                        "Analyze the following transcription of a customer phone call. "
                        "Evaluate the sentiment, look for signs of vocal stress (based on the words used, hesitations, etc.), "
                        "and identify potential fraud indicators. You must reply strictly in JSON format with the following keys: "
                        "sentiment (string), stress_level (Low/Medium/High), fraud_probability_score (number from 0 to 100), "
                        "and reasoning (brief string explaining why)."
                    ),
                },
                {
                    "role": "user",
                    "content": payload.transcription,
                },
            ],
        )
    except Exception as exc:  # pragma: no cover - external dependency
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Analysis service error: {exc}",
        ) from exc

    try:
        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            raise ValueError("Empty response from analysis service.")

        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object.")

        return parsed
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse analysis response: {exc}",
        ) from exc
