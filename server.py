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
MODEL = "gemini-2.0-flash-exp"  # Fastest model for audio

# Audio Settings
EXOTEL_SAMPLE_RATE = 8000
GEMINI_SAMPLE_RATE = 16000
# VAD Threshold: Audio RMS amplitude must be > this to be sent.
# 300-500 is a good range for phone lines to filter background hiss.
SILENCE_THRESHOLD = 400 

# Bot Persona & Tools
SYSTEM_INSTRUCTION = """
You are a helpful customer service assistant for a logistics company.
Keep your responses concise, under 2 sentences when possible.
Speak naturally and professionally.
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

if not API_KEY:
    logger.error("❌ GOOGLE_API_KEY not found in .env")
    sys.exit(1)

client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

# --- DATA MODELS (For Robustness) ---
class MediaFormat(BaseModel):
    encoding: str
    sample_rate: int
    bit_rate: str

class StartData(BaseModel):
    stream_sid: str
    call_sid: str
    media_format: MediaFormat

class MediaData(BaseModel):
    payload: str
    track: str
    chunk: Optional[str] = None
    timestamp: Optional[str] = None

class ExotelEvent(BaseModel):
    event: str
    stream_sid: Optional[str] = None
    start: Optional[StartData] = None
    media: Optional[MediaData] = None
    stop: Optional[Dict[str, Any]] = None

# --- AUDIO PROCESSING UTILS ---
def process_incoming_audio(payload: str) -> bytes:
    """Decodes Exotel Mu-Law 8k and converts to PCM 16k."""
    # 1. Decode Base64
    ulaw_data = base64.b64decode(payload)
    # 2. Convert Mu-Law to PCM 16-bit (8kHz)
    pcm_8k = audioop.ulaw2lin(ulaw_data, 2)
    # 3. Upsample 8kHz -> 16kHz for Gemini
    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
    return pcm_16k

def process_outgoing_audio(pcm_24k: bytes) -> str:
    """Downsamples Gemini PCM 24k to Exotel Mu-Law 8k."""
    # 1. Downsample 24kHz -> 8kHz
    pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
    # 2. Convert PCM 16-bit -> Mu-Law
    ulaw_data = audioop.lin2ulaw(pcm_8k, 2)
    # 3. Encode to Base64
    return base64.b64encode(ulaw_data).decode("utf-8")

# --- MAIN SERVER LOGIC ---
@app.get("/")
async def root():
    return {"status": "alive", "service": "Exotel-Gemini Relay"}

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ Exotel WebSocket Connected")
    
    # Session config
    session_config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": SYSTEM_INSTRUCTION
    }

    async with client.aio.live.connect(model=MODEL, config=session_config) as session:
        logger.info(f"🔹 Gemini Session Established: {MODEL}")
        
        # Task to handle incoming audio from Gemini -> Exotel
        async def send_to_exotel_task():
            try:
                while True:
                    async for response in session.receive():
                        if response.server_content and response.server_content.model_turn:
                            for part in response.server_content.model_turn.parts:
                                if part.inline_data:
                                    base64_payload = process_outgoing_audio(part.inline_data.data)
                                    msg = {
                                        "event": "media",
                                        "media": {
                                            "payload": base64_payload,
                                            "track": "outbound"
                                        }
                                    }
                                    await websocket.send_text(json.dumps(msg))
                        
                        if response.server_content and response.server_content.turn_complete:
                            logger.info("🔹 Gemini turn complete")
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in Gemini receive task: {e}")

        # Start the background task
        receive_task = asyncio.create_task(send_to_exotel_task())
        stream_sid = None

        try:
            while True:
                # 1. Receive data from Exotel
                data = await websocket.receive_text()
                
                # 2. Parse JSON safely
                try:
                    event_data = json.loads(data)
                    event = ExotelEvent(**event_data) # Pydantic validation
                except Exception:
                    # Ignore malformed packets to prevent crash
                    continue

                # 3. Handle Events
                if event.event == "start":
                    stream_sid = event.start.stream_sid
                    logger.info(f"📩 Call Started. Stream SID: {stream_sid}")
                
                elif event.event == "media" and event.media:
                    # Logic: Only send audio if it's loud enough (VAD)
                    try:
                        pcm_16k = process_incoming_audio(event.media.payload)
                        rms = audioop.rms(pcm_16k, 2)
                        
                        if rms > SILENCE_THRESHOLD:
                            await session.send(input={"data": pcm_16k, "mime_type": "audio/pcm"}, end_of_turn=False)
                        else:
                            # Silence detected - do nothing, let Gemini decide when to speak
                            # This prevents "breathing" noise from keeping the turn open
                            pass
                    except Exception as e:
                        logger.error(f"Error processing media: {e}")

                elif event.event == "stop":
                    logger.info("🛑 Exotel Stop Event Received")
                    break

        except WebSocketDisconnect:
            logger.warning("🔌 Exotel WebSocket Disconnected")
        except Exception as e:
            logger.error(f"⚠️ Critical Server Error: {e}")
        finally:
            receive_task.cancel()
            await websocket.close()
            logger.info("🔒 Connection Closed")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, log_level="warning")