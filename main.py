import os
import tempfile
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from openai import OpenAI

load_dotenv()

app = FastAPI()
client = OpenAI()


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
