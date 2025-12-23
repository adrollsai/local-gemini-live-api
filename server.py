import asyncio
import json
import base64
import audioop
import uvicorn
import logging
import os
import sys
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from pydantic import BaseModel, ValidationError
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
PORT = int(os.getenv("PORT", 5000))
API_KEY = os.getenv("GOOGLE_API_KEY")
# Using 2.0 Flash for low latency
MODEL = os.getenv("MODE")

# Audio Configuration
EXOTEL_SAMPLE_RATE = 8000
GEMINI_SAMPLE_RATE = 24000  # Gemini 2.0 Flash native output
INPUT_SAMPLE_RATE = 16000   # Gemini prefers 16k input
# VAD Threshold: 200 is very sensitive (picks up whispers), 
# preventing the "no sound" issue while still filtering empty static.
SILENCE_THRESHOLD = 200 

# Instructions for the bot
SYSTEM_INSTRUCTION = """
You are a helpful logistics customer service assistant. 
Your goal is to answer questions about delivery status and rates.
Keep your responses concise (1-2 sentences) to reduce latency on the phone line.
Speak naturally and professionally.
"""

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Exotel-Gemini")

app = FastAPI()

if not API_KEY:
    logger.error("❌ GOOGLE_API_KEY not found in .env file. Exiting.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})

# --- DATA MODELS (Restoring Robustness) ---
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

# --- AUDIO PROCESSING UTILITIES ---
def pcm_to_ulaw(pcm_data: bytes) -> str:
    """Converts PCM (16-bit, various rates) to Exotel compatible Mu-Law (8kHz)."""
    try:
        # Downsample from Gemini (24k) to Exotel (8k)
        pcm_8k, _ = audioop.ratecv(pcm_data, 2, 1, GEMINI_SAMPLE_RATE, EXOTEL_SAMPLE_RATE, None)
        # Convert Linear PCM to Mu-Law
        ulaw_data = audioop.lin2ulaw(pcm_8k, 2)
        return base64.b64encode(ulaw_data).decode("utf-8")
    except Exception as e:
        logger.error(f"Audio Output Conversion Error: {e}")
        return ""

def ulaw_to_pcm(ulaw_payload: str) -> bytes:
    """Converts Exotel Mu-Law (8kHz) to PCM (16kHz) for Gemini."""
    try:
        ulaw_data = base64.b64decode(ulaw_payload)
        pcm_8k = audioop.ulaw2lin(ulaw_data, 2)
        # Upsample from 8k to 16k for better recognition by Gemini
        pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, EXOTEL_SAMPLE_RATE, INPUT_SAMPLE_RATE, None)
        return pcm_16k
    except Exception as e:
        logger.error(f"Audio Input Conversion Error: {e}")
        return b""

# --- MAIN WEBSOCKET HANDLER ---
@app.get("/")
async def root():
    return {"status": "running", "service": "Exotel Gemini Relay"}

@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ Exotel WebSocket Connected")

    session_config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": SYSTEM_INSTRUCTION
    }

    try:
        async with client.aio.live.connect(model=MODEL, config=session_config) as session:
            logger.info(f"🔹 Connected to Gemini Session: {MODEL}")
            
            # --- FIX 3: WAKE UP MESSAGE ---
            # Force Gemini to speak immediately so you know the connection works.
            # 'end_of_turn=True' tells Gemini to reply.
            await session.send(input="Hello, introduce yourself.", end_of_turn=True)
            logger.info("👋 Sent wake-up trigger to Gemini")

            # --- OUTBOUND TASK (Gemini -> Exotel) ---
            async def send_audio_to_exotel():
                try:
                    while True:
                        async for response in session.receive():
                            # Handle Audio
                            if response.server_content and response.server_content.model_turn:
                                for part in response.server_content.model_turn.parts:
                                    if part.inline_data:
                                        payload = pcm_to_ulaw(part.inline_data.data)
                                        if payload:
                                            msg = {
                                                "event": "media",
                                                "media": {
                                                    "payload": payload,
                                                    "track": "outbound"
                                                }
                                            }
                                            await websocket.send_text(json.dumps(msg))
                            
                            # Handle Turn Completion (Optional: clear buffers)
                            if response.server_content and response.server_content.turn_complete:
                                # logger.info("🔹 Bot finished speaking") 
                                pass

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in outbound task: {e}")
                    traceback.print_exc()

            # Start the outbound background task
            outbound_task = asyncio.create_task(send_audio_to_exotel())

            # --- INBOUND LOOP (Exotel -> Gemini) ---
            stream_sid = None
            try:
                while True:
                    data = await websocket.receive_text()
                    
                    try:
                        event_data = json.loads(data)
                        # Pydantic validation for robustness
                        event = ExotelEvent(**event_data)
                    except ValidationError as ve:
                        logger.warning(f"Invalid Event format: {ve}")
                        continue
                    except json.JSONDecodeError:
                        continue

                    if event.event == "start":
                        stream_sid = event.start.stream_sid
                        logger.info(f"📩 Stream Started: {stream_sid}")

                    elif event.event == "media" and event.media:
                        # 1. Process Audio
                        pcm_audio = ulaw_to_pcm(event.media.payload)
                        
                        # 2. VAD Check (Fix for Latency & Silence)
                        rms = audioop.rms(pcm_audio, 2)
                        
                        if rms > SILENCE_THRESHOLD:
                            # --- FIX 1: NO LOGGING HERE ---
                            # Sending audio silently to avoid spamming logs
                            await session.send(input={"data": pcm_audio, "mime_type": "audio/pcm"}, end_of_turn=False)
                        else:
                            # Silence detected. We ignore it to prevent Gemini from 
                            # waiting 5s for you to "finish" a sentence of silence.
                            pass

                    elif event.event == "stop":
                        logger.info("🛑 Stop Event Received")
                        break

            except WebSocketDisconnect:
                logger.info("🔌 WebSocket Disconnected from Exotel")
            except Exception as e:
                logger.error(f"⚠️ Error in inbound loop: {e}")
            finally:
                outbound_task.cancel()
                logger.info("🔒 Closing session")

    except Exception as e:
        logger.error(f"🔥 Critical Connection Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)