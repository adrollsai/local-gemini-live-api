import asyncio
import json
import base64
import audioop
import uvicorn
import logging
import os
import sys
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
PORT = int(os.getenv("PORT", 5000))
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = os.getenv("MODEL")

# Audio Settings
# Exotel is 8000Hz mulaw, Gemini is 24000Hz PCM (usually)
EXOTEL_SAMPLE_RATE = 8000
GEMINI_SAMPLE_RATE = 24000 
# Lowered threshold slightly to ensure it picks up quieter phone lines
SILENCE_THRESHOLD = 300 

SYSTEM_INSTRUCTION = """
You are a helpful logistics assistant. 
When the conversation starts, greet the user briefly.
Keep your responses concise (1-2 sentences). 
Speak naturally and professionally.
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

if not API_KEY:
    logger.error("❌ GOOGLE_API_KEY not found in .env")
    sys.exit(1)

client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

# --- DATA MODELS ---
class MediaData(BaseModel):
    payload: str
    track: Optional[str] = None
    chunk: Optional[str] = None
    timestamp: Optional[str] = None

class StartData(BaseModel):
    stream_sid: str
    call_sid: str

class ExotelEvent(BaseModel):
    event: str
    start: Optional[StartData] = None
    media: Optional[MediaData] = None
    stream_sid: Optional[str] = None

# --- AUDIO UTILS ---
def process_incoming_audio(payload: str) -> bytes:
    """Decodes Exotel Mu-Law 8k -> PCM 16k (Gemini Input)"""
    ulaw_data = base64.b64decode(payload)
    pcm_8k = audioop.ulaw2lin(ulaw_data, 2)
    # Gemini usually expects 16k input for best results, though 24k is output
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return pcm_16k

def process_outgoing_audio(pcm_24k: bytes) -> str:
    """Downsamples Gemini PCM 24k -> Exotel Mu-Law 8k"""
    # Convert 24k PCM to 8k PCM
    pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
    # Convert 8k PCM to Mu-Law
    ulaw_data = audioop.lin2ulaw(pcm_8k, 2)
    return base64.b64encode(ulaw_data).decode("utf-8")

# --- SERVER ---
@app.get("/")
async def root():
    return {"status": "alive"}

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ Exotel WebSocket Connected")
    
    session_config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": SYSTEM_INSTRUCTION
    }

    async with client.aio.live.connect(model=MODEL, config=session_config) as session:
        logger.info(f"🔹 Gemini Session Established: {MODEL}")
        
        # 1. Send Initial Prompt to force Gemini to speak first
        # This fixes the "Dead Air" issue where both parties wait
        await session.send(input="Hello, please introduce yourself briefly.", end_of_turn=True)
        logger.info("👋 Sent initial greeting trigger to Gemini")

        async def send_to_exotel_task():
            """Background task to receive audio from Gemini and send to Exotel"""
            try:
                async for response in session.receive():
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.inline_data:
                                # Process audio
                                try:
                                    base64_payload = process_outgoing_audio(part.inline_data.data)
                                    msg = {
                                        "event": "media",
                                        "media": {
                                            "payload": base64_payload,
                                            "track": "outbound"
                                        }
                                    }
                                    await websocket.send_text(json.dumps(msg))
                                except Exception as audio_err:
                                    logger.error(f"Audio processing error: {audio_err}")

                    if response.server_content and response.server_content.turn_complete:
                        # Log when Gemini finishes a sentence
                        # logger.info("🔹 Gemini turn complete")
                        pass
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error receiving from Gemini: {e}")

        # Start background task
        receive_task = asyncio.create_task(send_to_exotel_task())
        
        try:
            while True:
                data = await websocket.receive_text()
                
                try:
                    event_data = json.loads(data)
                    # Basic validation
                    event_type = event_data.get("event")
                except json.JSONDecodeError:
                    continue

                if event_type == "start":
                    sid = event_data.get("start", {}).get("stream_sid", "unknown")
                    logger.info(f"📩 Call Started. Stream SID: {sid}")
                
                elif event_type == "media":
                    media_payload = event_data.get("media", {}).get("payload")
                    if media_payload:
                        try:
                            pcm_16k = process_incoming_audio(media_payload)
                            
                            # VAD Check
                            rms = audioop.rms(pcm_16k, 2)
                            if rms > SILENCE_THRESHOLD:
                                # Send audio to Gemini
                                await session.send(input={"data": pcm_16k, "mime_type": "audio/pcm"}, end_of_turn=False)
                            else:
                                # Silence - do nothing
                                pass
                        except Exception as e:
                            logger.error(f"Error processing incoming media: {e}")

                elif event_type == "stop":
                    logger.info("🛑 Exotel Stop Received")
                    break

        except WebSocketDisconnect:
            logger.info("🔌 Exotel Disconnected")
        except Exception as e:
            logger.error(f"⚠️ Critical Error: {e}")
        finally:
            receive_task.cancel()
            logger.info("🔒 Connection Closed")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="info")